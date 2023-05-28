from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import xgboost
from sklearn import metrics
from random import randint
import numpy as np
import copy
# import ViennaRNA


aptamerPath = "combined_sequences"
sequences = "cleaned_data.csv"
amt = 250
total = 0

df = []
dnaSequences = []
aptaProperties = dict()
dnaProperties = []
knownIndexes = []

rf = RandomForestClassifier() # RandomForest
xgb = xgboost.XGBClassifier() # XGBoost
svm1 = svm.SVC(C=10,gamma=0.01) # modified SVM
svm2 = svm.SVC() # unmodified SVM

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

with open("aptamers12.txt", "r+") as aptamers, open("NDB_cleaned_1.txt", "r+") as dnas, open("cleaned_data.csv", "r+") as csv:
    aptamerData = aptamers.read().split("\n")
    dnaData = dnas.read().split("\n")
    labels = csv.read().split(",")
    dnaData = list(filter(lambda x: len(x) > 0, dnaData))
    aptamerData = list(filter(lambda x: len(x) > 0, aptamerData))
    # use hstack to add the computed feature matrix (melting point) to the countvectorizer result.
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(6, 6), lowercase=False)
    dnaSequences = dnaData + aptamerData
    for i in dnaSequences:
        dnaProperties.append([computeMT(i)])
    amt = len(aptamerData)
    while len(knownIndexes) < len(dnaData):
        current_round_count = 0
        dnaSequences2 = copy.deepcopy(dnaSequences)
        while current_round_count < amt:
            i = randint(0, len(dnaData) - 1)
            if i not in knownIndexes:
                dnaSequences2.append(dnaData[i])
                knownIndexes.append(i)
                current_round_count += 1
            if len(knownIndexes) == len(dnaData):
                break
        vectorizer = vectorizer.fit(dnaSequences2)
    X = vectorizer.transform(dnaSequences)
    # this is how to add a thing onto the end of each member list of a sparse matrix.
    X = np.column_stack((X.toarray(), dnaProperties)) # type: ignore
    dnaProperties = np.hstack(dnaProperties)
    df = [0] * len(dnaData) + [1] * len(aptamerData)
    results = [[], [], [], []]
    resDict = dict()
    for _ in range(10):
        x_train, x_test, y_train, y_test = train_test_split(X, df)
        rf.fit(x_train, y_train)
        xgb.fit(x_train, y_train)
        svm1.fit(x_train, y_train)
        svm2.fit(x_train, y_train)
        rf_y_pred = rf.predict(x_test)
        xgb_y_pred = xgb.predict(x_test)
        svm1_y_pred = svm1.predict(x_test)
        svm2_y_pred = svm2.predict(x_test)
        results[0].append(metrics.balanced_accuracy_score(y_true=y_test, y_pred=rf_y_pred))
        results[1].append(metrics.balanced_accuracy_score(y_true=y_test, y_pred=xgb_y_pred))
        results[2].append(metrics.balanced_accuracy_score(y_true=y_test, y_pred=svm1_y_pred))
        results[3].append(metrics.balanced_accuracy_score(y_true=y_test, y_pred=svm2_y_pred))
    resDict["RF"] = tuple(results[0])
    resDict["XGBoost"] = tuple(results[1])
    resDict["SVM1"] = tuple(results[2])
    resDict["SVM2"] = tuple(results[3])
    for k, v in resDict.items():
        print(k, ":", np.average(v), np.std(v))

    

