# this is for the results of the Aptagen Spider. It only keeps sequences.

filePath = "aptagen_sequences_dna_only.txt"

numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."]
bases = ["A", "T", "C", "G", "U"]

aptamers = list()
sequences = list()

def strToNum(s: str) -> float:
    ret = ""
    for i in s:
        if i in numbers:
            ret += i
    return float(ret)

with open(filePath, "r+", encoding="utf-8") as f:
    data = f.read().split(", ")
    index = 0
    startTags = []
    while index < len(data):
        if "5'" in data[index]:
            try:
                index += 1
                seq = ""
                while index < len(data) and data[index] != "3'":
                    seq += data[index]
                    index += 1
                sequences.append(seq)
            except IndexError:
                print(index)
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

for k in aptamers:
    print(k)
