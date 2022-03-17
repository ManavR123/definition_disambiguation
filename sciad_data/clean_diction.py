import json
import re

with open("sciad_data/diction.json") as f:
    acronym_to_expansion = json.load(f)

for acronym in acronym_to_expansion:
    cleaned_expansions = []
    for expansion in acronym_to_expansion[acronym]:
        cleaned = (
            re.sub(r'\s([?.!,\'"](?:\s|$))', r"\1", expansion)
            .replace(" - ", "-")
            .replace(" 's", "'s")
            .replace(" / ", "/")
        )
        cleaned_expansions.append(cleaned)
    acronym_to_expansion[acronym] = cleaned_expansions

with open("sciad_data/diction.json", "w") as f:
    json.dump(acronym_to_expansion, f, indent=2)
