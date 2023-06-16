from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.model_selection import cross_validate
import xgboost
import sklearn.metrics
from random import randint
import numpy as np
import copy
import RNA
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sys import exit

aptamerPath = "combined_sequences"
sequences = "cleaned_data.csv"
amt = 250
total = 0
do_param_search = False
df = []
dnaSequences = []
aptaProperties = dict()
dnaProperties = []
validationProperties = []


earlyExit = True

def computeMT(seq: str) -> float:
    sequenceLength = len(seq)
    adenosine = 0
    cytosine = 0
    guanosine = 0
    thymine = 0
    for s in seq:
        match s:
            case "A":
                adenosine += 1
            case "C":
                cytosine += 1
            case "G":
                guanosine += 1
            case "T":
                thymine += 1
    if sequenceLength < 14:
        return (adenosine + thymine) * 2 + (guanosine + cytosine) * 4
    else:
        return 64.9 + 41 * (guanosine + cytosine - 16.4) / (adenosine + thymine + guanosine + cytosine)

def preprocessing(sequences: list[str], vectorizer, special: int, max_features=None):
    knownIndexes = []
    amt = len(sequences)
    while len(knownIndexes) < special:
        current_round_count = 0
        dnaSequences2 = copy.deepcopy(dnaSequences)
        while current_round_count < amt:
            i = randint(0, special - 1)
            if i not in knownIndexes:
                dnaSequences2.append(sequences[i])
                knownIndexes.append(i)
                current_round_count += 1
            if len(knownIndexes) == special:
                break
        vectorizer = vectorizer.fit(dnaSequences2)
    return vectorizer.transform(sequences)

with open("structures3.txt", "r+") as structures2, open("dna_aptamers.txt", "r+") as valset, open("aptamers12.txt", "r+") as aptamers, open("NDB_cleaned_1.txt", "r+") as dnas, open("cleaned_data.csv", "r+") as csv, open("structures.txt", "r+") as structures:
    # rewrite as a function that takes a file or string list, so I don't have to keep adding more vectorizers.
    aptamerData = aptamers.read().split("\n")
    dnaData = dnas.read().split("\n")
    labels = csv.read().split(",")
    validation_set = valset.read().split("\n")
    validation_set = list(filter(lambda x: len(x) > 0, validation_set))
    dnaData = list(filter(lambda x: len(x) > 0, dnaData))
    aptamerData = list(filter(lambda x: len(x) > 0, aptamerData))
    structList = structures.read().split("\n")
    structList = list(filter(lambda x: len(x) > 0, structList))
    structList2 = structures2.read().split("\n")
    structList2 = list(filter(lambda x: len(x) > 0, structList2))
    
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(6, 6), lowercase=False)
    ssVectorizer = CountVectorizer(analyzer="char", ngram_range=(6, 6))
    dnaSequences = dnaData + aptamerData
    for i in dnaSequences:
        ss, mfe = RNA.fold(i)
        dnaProperties.append([computeMT(i), mfe])
    for i in validation_set:
        ss, mfe = RNA.fold(i)
        validationProperties.append([computeMT(i), mfe])
    print("sequence properties computed")
    X = preprocessing(dnaSequences, vectorizer, len(dnaData))
    vectorizer2 = CountVectorizer(analyzer="char", ngram_range=(6, 6), lowercase=False, max_features=len(vectorizer.get_feature_names_out()))
    X2 = preprocessing(validation_set, vectorizer2, len(validation_set))
    structs = preprocessing(structList, ssVectorizer, len(structList))
    ssVectorizer2 = vectorizer = CountVectorizer(analyzer="char", ngram_range=(6, 6), max_features=len(ssVectorizer.get_feature_names_out()))
    structs2 = preprocessing(structList2, ssVectorizer2, len(structList2))
    
    print("vectorization complete")

    # this is how to add a thing onto the end of each member list of a sparse matrix.
    X = np.column_stack((X.toarray(), structs.toarray())) # type: ignore
    X = np.column_stack((X, dnaProperties)) # type: ignore
    # dnaProperties = np.hstack(dnaProperties)
    X2 = np.column_stack((X2.toarray(), structs2.toarray())) # type: ignore
    # validationProperties = np.hstack(validationProperties)
    X2 = np.column_stack((X2, validationProperties))

    df = [0] * len(dnaData) + [1] * len(aptamerData)
    df2 = [1] * len(validation_set)

    best_weight = round(len(dnaData) / len(aptamerData))
    print("starting algorithm training")
    xgb2 = xgboost.XGBClassifier(scale_pos_weight=best_weight)
    cv_runs = 20
    cvs = cross_validate(xgb2, X, df, cv=cv_runs, scoring='balanced_accuracy', return_train_score=True)
    print("finished training, printing results")

    xgb2.fit(X, df)
    prediction = xgb2.predict(X2)
    print(sklearn.metrics.accuracy_score(y_pred=prediction, y_true=df2))
