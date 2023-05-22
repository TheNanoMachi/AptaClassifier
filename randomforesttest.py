from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

aptamerPath = "combined_sequences"
sequences = "cleaned_data.csv"
amt = 250
total = 0

df = []
dnaSequences = []
aptaProperties = dict()
dnaProperties = []
knownIndexes = []

rf = RandomForestClassifier()

with open("aptamers12.txt", "r+") as aptamers, open("NDB_cleaned_1.txt", "r+") as dnas, open("cleaned_data.csv", "r+") as csv:
    aptamerData = aptamers.read().split("\n")
    dnaData = dnas.read().split("\n")
    labels = csv.read().split(",")
    dnaData = list(filter(lambda x: len(x) > 0, dnaData))
    aptamerData = list(filter(lambda x: len(x) > 0, aptamerData))
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(6, 6), lowercase=False)
    dnaSequences = dnaData + aptamerData
    X = vectorizer.fit_transform(dnaSequences)
    df = [0] * len(dnaData) + [1] * len(aptamerData)
    x_train, x_test, y_train, y_test = train_test_split(X, df)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    print(balanced_accuracy_score(y_true=y_test, y_pred=y_pred))
    print(rf.score(x_test, y_test))