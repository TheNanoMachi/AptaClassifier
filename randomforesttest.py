from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
import xgboost
from sklearn import metrics
from random import randint
import numpy as np
import copy
import RNA
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

aptamerPath = "combined_sequences"
sequences = "cleaned_data.csv"
amt = 250
total = 0
do_param_search = False
df = []
dnaSequences = []
aptaProperties = dict()
dnaProperties = []
knownIndexes = []

# rf = RandomForestClassifier() # RandomForest
# scale_pos_weight = total_negative_examples / total_positive_examples
# xgb = xgboost.XGBClassifier(scale_pos_weight=20) # XGBoost
# svm1 = svm.SVC(C=10,gamma=0.01) # modified SVM
# svm2 = svm.SVC() # unmodified SVM

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
        _, mfe = RNA.fold(i)
        dnaProperties.append([computeMT(i), mfe])
    print("sequence properties computed")
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
    if do_param_search:
        print("Searching for best weights")
        weights = [1, 2, 5, 10, 20, 50, 100]
        params = dict(scale_pos_weight=weights)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
        xgb = xgboost.XGBClassifier()
        gridSearch = GridSearchCV(estimator=xgb, param_grid=params, n_jobs=None, cv=cv, scoring='roc_auc')
        grid_result = gridSearch.fit(X, df)
        print("Best Score: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, std, param in zip(means, stds, params):
            print("Mean: %f StDev: %f Config: %r" %(mean, std, param))
        best_weight = grid_result.best_params_['scale_pos_weight']
    else:
        best_weight = round(len(dnaData) / len(aptamerData))
    print("starting algorithm training")
    xgb2 = xgboost.XGBClassifier(scale_pos_weight=best_weight)
    cvs = cross_validate(xgb2, X, df, cv=10, scoring='balanced_accuracy', return_train_score=True)
    print("finished training, printing results")
    for k, v in cvs.items():
        if k == "fit_time" or k == "score_time":
            continue
        else:
            print(k, "average", np.mean(v), "standard deviation", np.std(v))
        print(k, v)
        plt.plot(range(len(v)), v, label=k)
    plt.legend()
    plt.show()
    

