from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from random import randint
import numpy as np

aptamerPath = "combined_sequences"
sequences = "cleaned_data.csv"
amt = 250
total = 0
classifier = svm.SVC(C=10,gamma=0.01)

df = []
dnaSequences = []
knownIndexes = []
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


with open("aptamers12.txt", "r+") as aptamers, open("NDB_cleaned_1.txt", "r+") as dnas, open("cleaned_data.csv", "r+") as csv:
    aptamerData = aptamers.read().split("\n")
    dnaData = dnas.read().split("\n")
    labels = csv.read().split(",")
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(6, 6), lowercase=False)
    dnaSequences += aptamerData
    allData = aptamerData + dnaData
    df += [1] * len(aptamerData)
    runNumber = 0
    print("run number\t accuracy")
    while len(df) < len(dnaData):
        current_round_count = 0
        while current_round_count < amt:
            i = randint(0, len(dnaData) - 1)
            if i not in knownIndexes:
                dnaSequences.append(dnaData[i])
                knownIndexes.append(i)
                current_round_count += 1
        x = vectorizer.fit_transform(dnaSequences)
        properties = generateFeatures(dnaSequences)
        df += [0] * amt
        x_train, x_test, y_train, y_test = train_test_split(x, df)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        print("run", runNumber, ": \t", metrics.balanced_accuracy_score(y_true=y_test, y_pred=y_pred))
        runNumber += 1
