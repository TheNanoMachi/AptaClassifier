with open("combined_sequences copy", "r+") as file, open("output.csv", "w+", encoding="utf-8") as output:
    data = list(filter(lambda x: len(x) > 0, file.read().split("\n")))
    for seq in data:
        adenosine = 0
        cytosine = 0
        glutamine = 0
        thymine = 0
        length = len(seq)
        for i in seq:
            match i:
                case "A":
                    adenosine += 1
                case "C":
                    cytosine += 1
                case "G":
                    glutamine += 1
                case "T":
                    thymine += 1
        output.write(seq + "," + str(float(adenosine) / length) + "," + str(float(cytosine) / length) + "," + str(float(glutamine) / length) + "," + str(float(thymine) / length) + "," + "\n")