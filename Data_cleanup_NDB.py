filePath = "NDB.txt"

bases = ["A", "T", "C", "G", "U"]
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

with open("NDB_cleaned.txt", "w+", encoding="utf-8") as file:
    for s in sequences:
        file.write(s)
        file.write("\n")
