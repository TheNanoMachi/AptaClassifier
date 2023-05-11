from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

aptamerPath = "combined_sequences.txt"
results = []

classifier = svm.SVC(C=10,gamma=0.01)
categories = ["0", "1"]

x_train, x_test, y_train, y_test = 0, 0, 0, 0

with open(aptamerPath, "r+", encoding="utf-8", errors="ignore") as file:
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(6, 6), lowercase=False)
    x = vectorizer.fit_transform(file)
    results += vectorizer.get_feature_names_out().tolist()
    x_train, x_test, y_train, y_test = train_test_split(results, train_size=0.8)
    # results = [i for i in results if "\n" not in i]
    classifier.fit(x_train, y_train)

# for i in results:
#     print(i)

with open("sequences.txt", "r+", encoding="utf-8") as file:
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(6, 6), lowercase=False)
    print(classifier.predict(vectorizer.fit_transform(file)))
