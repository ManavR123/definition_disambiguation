import json
import random
from collections import defaultdict

import pandas as pd
import wikipedia
from tqdm import tqdm

from wikipedia_parsing.utils import invalid_term, valid_sense

random.seed(42)
MAX_TERMS = 500

with open("wikipedia_parsing/categories.txt", "r") as f:
    categories = f.read().splitlines()

term_to_sense = defaultdict(list)
rows = []
seen_terms = set()
with open("wikipedia_parsing/ambiguous_terms_log.txt", "w") as f:
    for category in tqdm(categories):
        terms = wikipedia.page(category, auto_suggest=False).links
        for term in tqdm(terms):
            if term.lower() in seen_terms or invalid_term(term):
                continue

            seen_terms.add(term.lower())
            try:
                wikipedia.page(term, redirect=False, auto_suggest=False)
                wikipedia.page(f"{term} (disambiguation)", redirect=False, auto_suggest=False)
            except wikipedia.exceptions.DisambiguationError as e:
                options = set([op.lower() for op in e.options if valid_sense(op, term.split()[0])])
                for t in options:
                    try:
                        summary = wikipedia.summary(t, redirect=False, auto_suggest=False)
                        rows.append([t, " ".join(summary.replace("\n", " ").split())])
                        term_to_sense[term].append(t)
                        print(f"{term} -> {t}", file=f)
                    except (wikipedia.exceptions.RedirectError, wikipedia.exceptions.PageError) as e:
                        continue
            except (wikipedia.exceptions.PageError, wikipedia.exceptions.RedirectError):
                continue

            if len(term_to_sense) >= MAX_TERMS:
                break

print(f"Found {len(term_to_sense)} terms")
print(f"Found {len(rows)} senses")
pd.DataFrame(rows, columns=["term", "summary"]).to_csv("wikipedia_parsing/ambiguous_terms.tsv", sep="\t", index=False)
json.dump(term_to_sense, open("wikipedia_parsing/ambiguous_term_to_senses.json", "w"))
