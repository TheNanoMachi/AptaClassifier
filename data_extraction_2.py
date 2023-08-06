import re
import functools as ft

cleaned_data = []

with open("aptagen_sequences_dna_only-1.txt", "r+", encoding='utf-8') as sequences:
    data = list(filter(lambda x: len(x) > 0, sequences.read().split(" || ")))
    for aptamer in data:
        cleaned_data.append(list(map(lambda x: re.sub(r"\{\{ | \}\}", "", str(x)), re.findall(r"\{\{.*?\}\}", aptamer))))

with open("cleaned_aptamers_with_ref.txt", "w+") as output:
    for cleaned_aptamer in cleaned_data:
        output.write("|".join(cleaned_aptamer))
        output.write("\n")
    