import json
import random
from collections import defaultdict

import pandas as pd
import wikipedia
from tqdm import tqdm

random.seed(42)


ambiguous_terms = open("sense_distribution/ambiguous_terms.txt", "r").read().splitlines()
term_to_sense = defaultdict(list)
rows = []
for term in tqdm(ambiguous_terms):
    try:
        wikipedia.page(term, redirect=False, auto_suggest=False)
    except wikipedia.exceptions.DisambiguationError as e:
        options = set([op.lower() for op in e.options if filter(op, term.split()[0])])
        options = random.sample(options, k=min(len(options), 15))
        for t in options:
            try:
                summary = wikipedia.summary(t, redirect=True, auto_suggest=False)
                rows.append([t, " ".join(summary.replace("\n", " ").split())])
                term_to_sense[term].append(t)
            except (wikipedia.exceptions.RedirectError, wikipedia.exceptions.PageError) as e:
                continue

print(f"Found {len(rows)} terms")
pd.DataFrame(rows, columns=["term", "summary"]).to_csv("sense_distribution/ambiguous_terms.tsv", sep="\t", index=False)
json.dump(term_to_sense, open("sense_distribution/ambiguous_term_to_senses.json", "w"))
