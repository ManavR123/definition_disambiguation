import argparse
import json

from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from indexing.utils_index import search


def get_examples(expansion, para, para_only=False):
    if para_only:
        if expansion.lower() in para.lower():
            return [para]
        return []

    examples = []
    para_sents = sent_tokenize(para)
    for sent in para_sents:
        if expansion.lower() in sent.lower():
            examples.append(sent)
    return examples


def main(args):
    with open("sciad_data/diction.json") as f:
        diction = json.load(f)

    all_expansions = []
    for exps in diction.values():
        all_expansions.extend(exps)

    expansion_to_sents = {}
    print("Collecting examples sentences...")
    for expansion in tqdm(all_expansions):
        examples = [expansion]
        results = search(expansion, ["pdf_parse"], 1000)
        for result in results:
            for para in result.get("pdf_parse", []):
                examples.extend(get_examples(expansion, para, para_only=args.para_only))
        expansion_to_sents[expansion] = examples

    fname = "paras" if args.para_only else "sents"
    with open(f"sciad_data/expansions_to_{fname}.json", "w") as f:
        json.dump(expansion_to_sents, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Search for examples for each expansion")
    parser.add_argument("--para_only", action="store_true", help="Only search in paragraphs")
    args = parser.parse_args()
    main(args)
