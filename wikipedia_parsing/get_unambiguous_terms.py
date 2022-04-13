import pandas as pd
from tqdm import tqdm
import wikipedia
from elasticsearch import Elasticsearch

from wikipedia_parsing.utils import invalid_term

es = Elasticsearch(hosts=["http://localhost:9200"], timeout=60, retry_on_timeout=True)
csi = "\x1B["
red = csi + "31;1m"
green = csi + "32;1m"
end = csi + "0m"


# get the glossary for chemistry terms
def get_summary(term):
    try:
        wikipedia.page(f"{term} (disambiguation)", redirect=False, auto_suggest=False)
        return False
    except wikipedia.exceptions.PageError:
        pass
    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.RedirectError):
        return False
    except Exception as e:
        print(e)
        return False
    try:
        return wikipedia.summary(term, redirect=False, auto_suggest=False)
    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.RedirectError):
        return False
    except Exception as e:
        print(e)
        return False


with open("wikipedia_parsing/categories.txt", "r") as f:
    categories = f.read().splitlines()

rows = []
seen_terms = set()
with open("wikipedia_parsing/log.txt", "w") as f:
    for category in tqdm(categories):
        terms = wikipedia.page(category, auto_suggest=False).links
        for term in tqdm(terms):
            if term.lower() in seen_terms or invalid_term(term):
                continue
            seen_terms.add(term.lower())
            summary = get_summary(term)
            if not summary:
                continue

            query = {
                "query": {
                    "multi_match": {
                        "query": f'"{term}"',
                        "fields": ["abstract"],
                        "type": "phrase",
                    }
                }
            }
            term_count = es.count(body=query, index="s2orc")["count"]
            if term_count >= 100:
                print(green + "Accepted:" + end, term, term_count, file=f, flush=True)
                summary = " ".join(summary.replace("\n", " ").split())
                rows.append([term, category, term_count, summary])
            else:
                print(red + "Rejected:" + end, term, term_count, file=f, flush=True)

print(f"Found {len(rows)} terms")
pd.DataFrame(rows, columns=["term", "category", "count", "summary"]).to_csv(
    "pseudowords/terms.tsv", sep="\t", index=False
)
