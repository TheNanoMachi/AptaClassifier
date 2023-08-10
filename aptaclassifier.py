from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import xgboost
from random import randint
import copy
import RNA
import pandas as pd
import time
import re

# program flags
earlyExit = True
permute_importances = True
calculate_importances = False
vectorize = False

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

def computeProperties(sequences: list[str]) -> list:
    sequence_properties = []
    for i in sequences:
        ss, mfe = RNA.fold(i)
        sequence_properties.append([computeMT(i), mfe] + countLoops(ss))
    return sequence_properties

def preprocessing(sequences: list[str], vec: CountVectorizer, special: int):
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

with open("validated.txt", "r+") as valset, open("aptamers12.txt", "r+") as aptamers, open("NDB_cleaned_1.txt", "r+") as dnas:
    dnaProperties = []
    validationProperties = []
    program_start = time.time()

    validation_set = list(filter(lambda x: len(x) > 0, valset.read().split("\n")))
    
    dnaData = list(filter(lambda x: len(x) > 0, dnas.read().split("\n")))

    aptamerData = list(filter(lambda x: len(x) > 0, aptamers.read().split("\n")))

    dnaSequences = dnaData + aptamerData + validation_set

    for i in dnaSequences:
        ss, mfe = RNA.fold(i)
        dnaProperties.append([computeMT(i), mfe] + countLoops(ss))

    print("sequence properties computed")

    column_labels = ["mp", "mfe", "hairpins", "internal_loops"]

    X = pd.DataFrame(dnaProperties, columns=column_labels)

    y = [0] * len(dnaData) + [1] * (len(aptamerData) + len(validation_set))

    print("starting algorithm training")
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.6)

    best_weight = round(len(dnaData) / (len(aptamerData) + len(validation_set)))

    xgb2 = xgboost.XGBClassifier(scale_pos_weight=best_weight)

    xgb2.fit(x_train, y_train)

    print("finished training, printing results")

    print("validated (true aptamers) results:")

    features = list(map(lambda x: pd.DataFrame(computeProperties([x]), columns=column_labels), validation_set))

    results = list(map(lambda x: xgb2.predict_proba(x), features))

    for i in range(len(results)):
        result = 0 if results[i][0][0] > results[i][0][1] else 1
        print(validation_set[i], results[i][0][0], results[i][0][1], result)

    print(f"time elapsed: {(time.time() - program_start): .3f} seconds")



