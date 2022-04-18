import json
from collections import defaultdict

import pandas as pd
import wikipedia
from tqdm import tqdm

from wikipedia_parsing.utils import invalid_term, valid_sense


with open("wikipedia_parsing/categories.txt", "r") as f:
    categories = f.read().splitlines()

term_to_sense = defaultdict(list)
rows = []
seen_terms = set()
with open("wikipedia_parsing/ambiguous_terms_log.txt", "w") as f:
    for category in tqdm(categories):
        count = 0
        terms = wikipedia.page(category, auto_suggest=False).links
        for term in tqdm(terms):
            if term.lower() in seen_terms or invalid_term(term):
                continue

            seen_terms.add(term.lower())
            try:
                wikipedia.page(term, redirect=False, auto_suggest=False)
                wikipedia.page(f"{term} (disambiguation)", redirect=False, auto_suggest=False)
            except wikipedia.exceptions.DisambiguationError as e:
                options = set([op.lower() for op in e.options if valid_sense(op, term)])
                if "(disambiguation)" in e.title:
                    options.add(term)

                for t in options:
                    try:
                        summary = wikipedia.summary(t, redirect=False, auto_suggest=False)
                        rows.append([t, " ".join(summary.replace("\n", " ").split())])
                        term_to_sense[term].append(t)
                        print(f"{len(term_to_sense)}: {term} -> {t}", file=f)
                    except wikipedia.exceptions.WikipediaException:
                        continue
                    except Exception as e:
                        print(e)
                        continue
            except (wikipedia.exceptions.PageError, wikipedia.exceptions.RedirectError):
                continue
            except Exception as e:
                print(e)
                continue

            if len(term_to_sense[term]) == 1:
                del term_to_sense[term]
                rows.pop()
            else:
                count += 1

        print(f"{category}: {count}")

print(f"Found {len(term_to_sense)} terms")
print(f"Found {len(rows)} senses")
pd.DataFrame(rows, columns=["term", "summary"]).to_csv("wikipedia_parsing/ambiguous_terms.tsv", sep="\t", index=False)
json.dump(term_to_sense, open("wikipedia_parsing/ambiguous_term_to_senses.json", "w"))
