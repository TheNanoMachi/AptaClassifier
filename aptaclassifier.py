from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2, RFECV
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, cross_validate, GridSearchCV, StratifiedShuffleSplit, learning_curve, StratifiedKFold
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
from functools import reduce
import re

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

# regex for finding loops: \(\.*\)

def countLoops(structure: str) -> list[int]:
    # hairpins
    hairpins = len(re.findall(r'\(\.*\)', structure))

    # internal loops
    forwardMatches =  re.findall(r"(?=(\(\.+\())", structure)
    reverseMatches =  re.findall(r"(?=(\)\.+\)))", structure)

    internal_loops = 0

    for i in forwardMatches:
        for j in reverseMatches:
            if len(i) == len(j):
                internal_loops += 1
                break

    return [hairpins, internal_loops] # type: ignore

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
def pack_and_sort_descending(labels: list[str], values: list[float] | list[int], threshold: float) -> list[list[str] | list[float] | list[int]]:
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

def count_features(feature: str, appearances: list[list[str]]) -> int:
    result = 0
    for appearance_list in appearances:
        for name in appearance_list:
            if feature == name:
                result += 1
    return result

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
        dnaProperties.append([computeMT(i), mfe] + countLoops(ss))
        # dnaProperties.append([mfe])
        sL.append(ss)
    for i in v_test:
        ss, mfe = RNA.fold(i)
        sL2.append(ss)
        validationProperties.append([computeMT(i), mfe] + countLoops(ss))
        # validationProperties.append([mfe])
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
    X = np.column_stack((X, dnaProperties))
    X2 = np.column_stack((X2.toarray(), structs2)) # type: ignore
    X2 = np.column_stack((X2, validationProperties))

    # X = np.column_stack((X.toarray(), dnaProperties))
    # X2 = np.column_stack((X2.toarray(), validationProperties))

    column_labels = vectorizer3.get_feature_names_out().tolist()
    column_labels += ssV3.get_feature_names_out().tolist()
    column_labels += ["mp", "mfe", "hairpins", "internal_loops"]
    # column_labels += ["mfe"]

    X = pd.DataFrame(X, columns=column_labels)    

    y = [0] * len(dnaData) + [1] * (len(aptamerData) + len(v_train))
    y2 = [1] * len(v_test)

    print("starting algorithm training")
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.6)

    eval_metric = ["auc", "logloss", "error"]

    best_weight = round(len(dnaData) / (len(aptamerData) + len(v_train)))

    xgb2 = xgboost.XGBClassifier(scale_pos_weight=best_weight, eval_metric=eval_metric)

    scoring = ["balanced_accuracy"]

    xgb2.fit(x_train, y_train)

    print("finished training, printing results")
  
    y_pred = xgb2.predict(x_test)

    print(f"accuracy: {balanced_accuracy_score(y_pred=y_pred, y_true=y_test): .5f}")

    booster = xgb2.get_booster()

    importance_types = ['weight', 'gain', 'total_gain', 'total_cover']

    scores = dict()

    for importance_type in importance_types:
        scores[importance_type] = booster.get_score(importance_type=importance_type)

    feature_names = []

    scoring = ['balanced_accuracy', 'f1', 'roc_auc']

    permute_importances = True

    if permute_importances:
        results = permutation_importance(xgb2, x_train, y_train, n_repeats=3, 
                                         max_samples=0.4, scoring=scoring, n_jobs=4)

        for result in results.values():
            feature_names.append(pack_and_sort_descending(column_labels, result.importances_mean, 0))
    else:
        results = dict()

    for score in scores.values():
        feature_names.append(pack_and_sort_descending(list(score.keys()), list(score.values()), 0))
    
    # top_n = min(list(map(lambda x: len(x[0]), feature_names)))

    top_appearances = []

    for feature in feature_names:
        # names = []
        # for i in range(top_n):
        #     names.append(feature[0][i])
        top_appearances += [feature[0]]

    # print(top_appearances)
    
    print(", ".join(list(scores.keys()) + list(results.keys())))
    # print(list(reduce(lambda x, y: x & y, (set(i) for i in top_appearances))))

    features_found = list(reduce(lambda x, y: x | y, (set(i) for i in top_appearances)))

    frequencies = list(map(lambda x: count_features(x, top_appearances), features_found))

    freq_dict = pack_and_sort_descending(features_found, frequencies, 0)

    for i in range(len(freq_dict[0])):
        print(freq_dict[0][i], freq_dict[1][i])

    # no mp: 0.97690
    # mp: 0.97709~0.98474

    # xgb_feature = pack_and_sort_descending(list(xgb_gain.keys()), list(xgb_gain.values()), 0.0001) # type: ignore   

    # shap_explainer = shap.Explainer(xgb2)
    # shap_test = shap_explainer(x_test)

    # print(shap_test.values.shape)
    # print(len(column_labels))

    # feature_values = []

    # for i in range(shap_test.values.shape[1]):
    #     feature = []
    #     for j in range(shap_test.values.shape[0]):
    #         feature.append(shap_test.values[j][i])
    #     feature_values.append(feature)

    # for i in range(len(feature_values)):
    #     if np.average(feature_values[i]) != 0:
    #         print(column_labels[i], np.absolute(np.average(feature_values[i])))
    
    # shap.plots.bar(shap_test)

    # xgb3 = xgboost.XGBClassifier(scale_pos_weight=best_weight)

    # cv = StratifiedKFold(5)

    # rfecv = RFECV(estimator=xgb3, step=500, cv=cv, scoring="balanced_accuracy", min_features_to_select=1)

    # print("created RFECV model")

    # rfecv.fit(X, y)

    # print(f"Optimal number of features: {rfecv.n_features_}")

    print(f"time elapsed: {(time.time() - program_start): .3f} seconds")



