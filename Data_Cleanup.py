filePath = "result3.txt"

numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."]
labels = ["Length:", "Molecular Weight:", "Extinction Coefficient:", "GC Content:"]
specialLabels = ["nmoles/OD", "μg/OD"]
bases = ["A", "T", "C", "G", "U"]

aptamers = dict()
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
        properties = list()
        for i in range(len(data)):
            if data[i] not in startTags and "==" in data[i]:
                j = i
                while "--" not in data[j]:
                    if data[j] in labels:
                        properties.append(strToNum(data[j + 1]))
                    elif data[j] in specialLabels:
                        properties.append(strToNum(data[j + 2]))
                    j += 1
                startTags.append(data[i])
                break
        adenosine = 0
        cytosine = 0
        glutamine = 0
        thymine = 0
        uracil = 0
        for i in sequence:
            match i:
                case "A":
                    adenosine += 1
                case "C":
                    cytosine += 1
                case "G":
                    glutamine += 1
                case "T":
                    thymine += 1
                case "U":
                    uracil += 1
        percentages = [adenosine , cytosine, glutamine, thymine, uracil]
        for i in range(len(percentages)):
            percentages[i] = round(percentages[i] / len(sequence), 4)
        aptamers[sequence] = tuple(properties + percentages)

for k, v in aptamers.items():
    print(k, v)
