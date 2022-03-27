import json
import re

import pandas as pd


def clean(text):
    return re.sub(r'\s([?.!,\'"](?:\s|$))', r"\1", text).replace(" - ", "-").replace(" 's", "'s")


acronym_to_expansion = json.load(open("sciad_data/diction.json"))
for acronym in acronym_to_expansion:
    cleaned_expansions = []
    for expansion in acronym_to_expansion[acronym]:
        cleaned = clean(expansion)
        cleaned_expansions.append(cleaned)
    acronym_to_expansion[acronym] = cleaned_expansions
with open("sciad_data/diction.json", "w") as f:
    json.dump(acronym_to_expansion, f, indent=2)

for file in ["train", "dev"]:
    fname = f"sciad_data/{file}.csv"
    df = pd.read_csv(fname)
    df["expansion"] = df["expansion"].apply(lambda x: clean(x))
    df.to_csv(fname, index=False)
