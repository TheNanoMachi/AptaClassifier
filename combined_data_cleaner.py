numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."]
bases = ["A", "T", "C", "G", "U"]

aptamers = list()
sequences = list()

results = "cleaned_data1.csv"
filePath = "proteins_raw.txt"

sequences = []

with open(filePath, "r+", encoding="utf-8") as file:
    dataDirty = file.read().split("\n")
    for row in dataDirty:
        sequence = ""
        chars = row.split(", ")
        for c in chars:
            for i in c:
                if i in bases:
                    sequence += i
        sequences.append(sequence.rstrip("A"))

with open(results, "a+", encoding="utf-8") as file:
    for s in sequences:
        file.write(s)
        file.write(",0,\n")

sequences = []
filePath = "aptagen_sequences_dna_only.txt"

with open(filePath, "r+", encoding="utf-8") as f:
    data = f.read().split(", ")
    index = 0
    while index < len(data):
        if "5'" in data[index]:
            index += 1
            seq = ""
            while index < len(data) and data[index] != "3'":
                seq += data[index]
                index += 1
            sequences.append(seq)
        index += 1
    for sequence in sequences:
        isSequence = True
        for i in sequence:
            if i not in bases:
                isSequence = False
                break
        if not isSequence:
            continue
        aptamers.append(sequence)

with open(results, "a+", encoding="utf-8") as file:
    for a in aptamers:
        file.write(a)
        file.write(",1,\n")