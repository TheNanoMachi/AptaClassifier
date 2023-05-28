from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from random import randint
import numpy as np
import copy

aptamerPath = "combined_sequences"
sequences = "cleaned_data.csv"
total = 0
classifier = svm.SVC(C=10,gamma=0.01)

df = []
dnaSequences = []
knownIndexes = []
properties = []
aptaProperties = dict()

def generateFeatures(data: list) -> dict:
    properties = dict()
    for seq in data:
        adenosine = 0
        cytosine = 0
        glutamine = 0
        thymine = 0
        length = len(seq)
        for i in seq:
            match i:
                case "A":
                    adenosine += 1
                case "C":
                    cytosine += 1
                case "G":
                    glutamine += 1
                case "T":
                    thymine += 1
        percentages = [float(adenosine) / length, float(cytosine) / length, float(glutamine) / length, float(thymine) / length]
        properties[seq] = tuple(percentages)
    return properties

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
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(6, 6), lowercase=False)
    dnaSequences += aptamerData
    allData = aptamerData + dnaData
    df += [1] * len(aptamerData)
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
    X = vectorizer.transform(allData)
    df += [0] * len(dnaData)
    x_train, x_test, y_train, y_test = train_test_split(X, df)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print(metrics.balanced_accuracy_score(y_pred=y_pred, y_true=y_test))
