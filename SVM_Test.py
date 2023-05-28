from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import re

aptamerPath = "output.csv"
df = pd.read_csv(aptamerPath)
sequences = "cleaned_data.csv"
classifier = svm.SVC(C=10,gamma=0.01)

classes = []

with open(sequences, "r+", encoding="utf-8") as csv:
    data = re.sub("^(,0,)", ",", csv.read(), flags=re.M).split(",")
    for i in data:
        if i == "1" or i == "0":
            classes.append(i)
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(6, 6), lowercase=False)
    x = vectorizer.fit_transform(df.loc[:,["Sequence"]].values.flatten())
    aux = df.loc[:,["A_ratio", "C_ratio", "G_ratio", "T_ratio"]].values
    x_train, x_test, y_train, y_test = train_test_split(aux, classes)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print(metrics.accuracy_score(y_pred=y_pred, y_true=y_test))
    print(metrics.balanced_accuracy_score(y_true=y_test, y_pred=y_pred))