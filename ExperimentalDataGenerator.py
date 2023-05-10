from sklearn.feature_extraction.text import CountVectorizer

aptamerPath = "sequences.txt"
nonAptamerPath = "NDB_cleaned.txt"
results = []

with open(aptamerPath, "r+", encoding="utf-8", errors="ignore") as fApta, open(nonAptamerPath, "r+", encoding="utf-8", errors="ignore") as fNonApta:
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(6, 6), lowercase=False)
    x = vectorizer.fit_transform(fApta)
    results += vectorizer.get_feature_names_out().tolist()
    y = vectorizer.fit_transform(fNonApta)
    results += vectorizer.get_feature_names_out().tolist()

results = [i for i in results if "\n" not in i]

for i in results:
    print(i)

