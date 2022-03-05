import json

from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from indexing.utils_index import search

with open("sciad_data/diction.json") as f:
    diction = json.load(f)

all_expansions = []
for exps in diction.values():
    all_expansions.extend(exps)

expansion_to_sents = {}
print("Collecting examples sentences...")
for expansion in tqdm(all_expansions):
    sents = [expansion]
    results = search(expansion, ["pdf_parse"], 1000)
    for result in results:
        for para in result.get("pdf_parse", []):
            para_sents = sent_tokenize(para)
            for sent in para_sents:
                if expansion.lower() in sent.lower():
                    sents.append(para)
    expansion_to_sents[expansion] = sents

with open("expansions_to_sents.json", "w") as f:
    json.dump(expansion_to_sents, f)