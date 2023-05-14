from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics

aptamerPath = "combined_sequences"
sequences = "cleaned_data.csv"
results = []

classifier = svm.SVC(C=10,gamma=0.01)

df = []

with open(sequences, "r+", encoding="utf-8") as csv:
    data = csv.read().split(",")
    for i in data:
        if i == "1" or i == "0":
            df.append(i)

with open(aptamerPath, "r+", encoding="utf-8", errors="ignore") as file:
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(6, 6), lowercase=False)
    x = vectorizer.fit_transform(file)
    x_train, x_test, y_train, y_test = train_test_split(x, df)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print(metrics.accuracy_score(y_pred=y_pred, y_true=y_test))
    print(metrics.balanced_accuracy_score(y_true=y_test, y_pred=y_pred))