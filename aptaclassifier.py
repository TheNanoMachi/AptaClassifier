from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, cross_validate, GridSearchCV, StratifiedShuffleSplit, learning_curve
import xgboost
from sklearn.ensemble import RandomForestClassifier
from random import randint
import numpy as np
import copy
import RNA
import matplotlib.pyplot as plt
from sys import exit

aptamerPath = "combined_sequences"
sequences = "cleaned_data.csv"
amt = 250
total = 0
do_param_search = False
y = []
dnaSequences = []
aptaProperties = dict()
dnaProperties = []
validationProperties = []


earlyExit = False

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
    sL = []
    sL2 = []
    
    v_train, v_test, pos_label1, pos_label2 = train_test_split(validation_set, 
                                                               [1] * len(validation_set), train_size=0.6)

    vectorizer = CountVectorizer(analyzer="char", ngram_range=(6, 6), lowercase=False)
    ssVectorizer = CountVectorizer(analyzer="char", ngram_range=(6, 6), lowercase=False)
    dnaSequences = dnaData + aptamerData + v_train
    for i in dnaSequences:
        ss, mfe = RNA.fold(i)
        dnaProperties.append([computeMT(i), mfe])
        sL.append(ss)
    for i in v_test:
        ss, mfe = RNA.fold(i)
        sL2.append(ss)
        validationProperties.append([computeMT(i), mfe])
    print("sequence properties computed")
    X = preprocessing(dnaSequences, vectorizer, len(dnaData))
    vectorizer2 = CountVectorizer(analyzer="char", ngram_range=(6, 6), lowercase=False, max_features=len(vectorizer.get_feature_names_out()))
    X2 = preprocessing(v_test, vectorizer2, len(v_test))
    structs = preprocessing(sL, ssVectorizer, len(sL))
    ssVectorizer2 = vectorizer = CountVectorizer(analyzer="char", ngram_range=(6, 6), lowercase=False, max_features=len(ssVectorizer.get_feature_names_out()))
    structs2 = preprocessing(sL2, ssVectorizer2, len(sL2))
    
    print("vectorization complete")
    # this is how to add a thing onto the end of each member list of a sparse matrix.
    print(X.shape)
    print(X2.shape)
    print(structs.shape)
    print(structs2.shape)
    structs = np.vstack(structs.toarray())
    structs2 = np.vstack(structs2.toarray())
    X = np.column_stack((X.toarray(), structs)) # type: ignore
    X = np.column_stack((X, dnaProperties)) # type: ignore
    X2 = np.column_stack((X2.toarray(), structs2)) # type: ignore
    X2 = np.column_stack((X2, validationProperties))

    print(X.shape)
    print(X2.shape)
    print(structs.shape)
    print(structs2.shape)

    if earlyExit:
        exit(0)

    y = [0] * len(dnaData) + [1] * (len(aptamerData) + len(v_train))
    y2 = [1] * len(v_test)

    best_weight = round(len(dnaData) / (len(aptamerData) + len(v_train)))
    print("starting algorithm training")

    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.6)

    eval_set = [(x_train, y_train), (x_test, y_test)]

    eval_metric = ["auc", "logloss", "error"]

    xgb2 = xgboost.XGBClassifier(scale_pos_weight=best_weight, eval_metric=eval_metric)
    rf = RandomForestClassifier()
    # cv_runs = 20
    # sss = StratifiedShuffleSplit(test_size=0.2, n_splits=cv_runs)
    # cvs = cross_validate(xgb2, X, y, cv=sss, scoring='balanced_accuracy', return_train_score=True, error_score='raise')
    # cvs2 = cross_validate(rf, X, y, cv=sss, scoring='balanced_accuracy', return_train_score=True)
    # sample_size, train_scores, test_scores = learning_curve(xgb2, X, y, cv=sss, scoring='balanced_accuracy') # type: ignore
    print("finished training, printing results")

    xgb = xgb2.fit(x_train, y_train, eval_set=eval_set)

    results = xgb2.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)

    # plt.plot(x_axis, results['validation_0']['auc'], label="train_auc")
    # plt.plot(x_axis, results['validation_0']['logloss'], label="train_logloss")
    # plt.plot(x_axis, results['validation_1']['auc'], label="test_auc")
    # plt.plot(x_axis, results['validation_1']['logloss'], label="test_logloss")
    plt.plot(x_axis, results['validation_0']['error'], label="train_error")
    plt.plot(x_axis, results['validation_1']['error'], label="test_error")
    # print(sample_size)
    # print(train_scores)
    # print(test_scores)
    # for i in range(5):
    #     plt.plot(range(len(train_scores[i])), train_scores[i], label="train_run_" + str(i))
    #     plt.plot(range(len(test_scores[i])), test_scores[i], label="test_run_" + str(i))
    # plt.plot(len(train_scores), train_scores, label="train_scores")
    # plt.plot(len(test_scores), test_scores, label="test_scores")
    # plt.plot(len(sample_size), sample_size, label="samples")
    # plt.ylim(0.9, 1)
    # plt.plot(range(len(cvs["test_score"])), cvs["test_score"], label="xgb_test_score")
    # plt.plot(range(len(cvs["train_score"])), cvs["train_score"], label="xgb_train_score")
    # plt.plot(range(len(cvs2["test_score"])), cvs2["test_score"], label="rf_test_score")
    # plt.plot(range(len(cvs2["train_score"])), cvs2["train_score"], label="rf_train_score")
    plt.legend()
    plt.show()
    

    # xgb2.fit(X, y)
    # prediction = xgb2.predict(X2)
    # print(sklearn.metrics.accuracy_score(y_pred=prediction, y_true=y2))
