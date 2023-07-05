from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, cross_validate, GridSearchCV, StratifiedShuffleSplit, learning_curve
import xgboost
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report, get_scorer_names
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import partial_dependence, permutation_importance
from random import randint
import numpy as np
import copy
import RNA
import matplotlib.pyplot as plt
from sys import exit
import pandas as pd
import time
import shap

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

def preprocessing(sequences: list[str], vec: CountVectorizer, special: int, max_features=None):
    knownIndexes = []
    amt = len(sequences)
    while len(knownIndexes) < special:
        current_round_count = 0
        dnaSequences2 = copy.deepcopy(sequences)
        while current_round_count < amt:
            i = randint(0, special - 1)
            if i not in knownIndexes:
                dnaSequences2.append(sequences[i])
                knownIndexes.append(i)
                current_round_count += 1
            if len(knownIndexes) == special:
                break
        vec = vec.fit(dnaSequences2)
    return vec.transform(sequences)

# take a list of labels and a list of values, and return a dictionary whose values are any values in values >= threshold
# and whose keys are their associated labels. labels and values should have the same length.
def pack_and_sort_descending(labels: list[str], values: list[float | int], threshold: float) -> list[list[str | float | int]]:
    important_values = []
    important_labels = []
    for i in range(len(values)):
        if values[i] >= threshold:
            important_values.append(values[i])
            important_labels.append(labels[i])
    package = zip(important_labels, important_values)
    package = sorted(package, reverse=True, key=lambda elem: elem[1])
    k = []
    v = []
    for i in package:
        k.append(i[0])
        v.append(i[1])
    return [k, v]

with open("structures3.txt", "r+") as structures2, open("dna_aptamers.txt", "r+") as valset, open("aptamers12.txt", "r+") as aptamers, open("NDB_cleaned_1.txt", "r+") as dnas, open("cleaned_data.csv", "r+") as csv, open("structures.txt", "r+") as structures:
    # rewrite as a function that takes a file or string list, so I don't have to keep adding more vectorizers.
    
    program_start = time.time()
    
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
    ssVectorizer2 = CountVectorizer(analyzer="char", ngram_range=(6, 6), lowercase=False, max_features=len(ssVectorizer.get_feature_names_out()))
    structs2 = preprocessing(sL2, ssVectorizer2, len(sL2))

    min_features = min(X.shape[1], X2.shape[1])

    vectorizer3 = CountVectorizer(analyzer="char", ngram_range=(6, 6), lowercase=False, max_features=min_features)
    
    X = preprocessing(dnaSequences, vectorizer3, len(dnaData))
    X2 = preprocessing(v_test, vectorizer3, len(v_test))

    min_features = min(structs.shape[1], structs2.shape[1])

    ssV3 = CountVectorizer(analyzer="char", ngram_range=(6, 6), lowercase=False, max_features=min_features)
    
    structs = preprocessing(sL, ssV3, len(sL))
    structs2 = preprocessing(sL2, ssV3, len(sL2))
    
    print("vectorization complete")
    # this is how to add a thing onto the end of each member list of a sparse matrix.

    structs = np.vstack(structs.toarray())
    structs2 = np.vstack(structs2.toarray())

    X = np.column_stack((X.toarray(), structs)) # type: ignore
    X = np.column_stack((X, dnaProperties)) # type: ignore
    X2 = np.column_stack((X2.toarray(), structs2)) # type: ignore
    X2 = np.column_stack((X2, validationProperties))

    column_labels = vectorizer3.get_feature_names_out().tolist()
    column_labels += ssV3.get_feature_names_out().tolist()
    column_labels += ["mp", "mfe"]

    X = pd.DataFrame(X, columns=column_labels)    

    y = [0] * len(dnaData) + [1] * (len(aptamerData) + len(v_train))
    y2 = [1] * len(v_test)

    print("starting algorithm training")
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.6)

    # eval_set = [(x_train, y_train), (x_test, y_test)]

    eval_metric = ["auc", "logloss", "error"]

    best_weight = round(len(dnaData) / (len(aptamerData) + len(v_train)))

    xgb2 = xgboost.XGBClassifier(scale_pos_weight=best_weight, eval_metric=eval_metric)

    scoring = ["balanced_accuracy"]

    xgb2.fit(x_train, y_train)

    # result = permutation_importance(xgb2, x_test, y_test, scoring=scoring, n_repeats=5, random_state=33)

    # importances = result[scoring[0]]["importances_mean"]

    print("finished training, printing results")

    print(f"time elapsed: {(time.time() - program_start): .3f} seconds")

    
    y_pred = xgb2.predict(x_test)

    print(f"accuracy: {balanced_accuracy_score(y_pred=y_pred, y_true=y_test): .5f}")

    # xgb_gain = xgb2.get_booster().get_score(importance_type='gain')
    # xgb_feature = pack_and_sort_descending(list(xgb_gain.keys()), list(xgb_gain.values()), 0.0001) # type: ignore
    shap_explainer = shap.Explainer(xgb2)
    shap_test = shap_explainer(x_test)
    
    shap.plots.bar(shap_test)
    # perm_importance = pack_and_sort_descending(column_labels, importances, 0.0001)
    
    # plt.subplot(1, 2, 1)
    # plt.bar(perm_importance[0], perm_importance[1])
    # plt.subplot(1, 2, 2)
    # fig = plt.figure()
    # plt.bar(xgb_feature[0], xgb_feature[1])
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    # fig.tight_layout()
    # plt.show()





