import argparse
import json
import re
import time
from collections import defaultdict

import pandas as pd
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from indexing.utils_index import search


def get_examples(expansion, para, para_only):
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


def get_examples_from_train(train_file, all_expansions):
    df = pd.read_csv(train_file).dropna(subset=["expansion", "text"])
    examples = defaultdict(list)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        expansion = row["expansion"]
        examples[expansion].extend(row["text"])

    assert all_expansions == set(examples), f"{all_expansions - set(examples)}"
    return examples


def get_examples_from_index(all_expansions, para_only):
    examples = {}
    for expansion in tqdm(all_expansions):
        examples = [expansion.lower()]
        results = search(expansion, ["pdf_parse"], 1000)
        for result in results:
            for para in result.get("pdf_parse", []):
                examples.extend(get_examples(expansion, para, para_only))
        examples[expansion] = examples


def main(args):
    diction = json.load(open("sciad_data/diction.json"))
    all_expansions = set().union(*diction.values())

    # add arguments to output
    expansion_to_sents = {f"arg-{k}": v for k, v in vars(args).items()}

    print("Collecting examples sentences...")
    if args.mode == "ExpansionOnly":
        examples = {expansion: [expansion.lower()] for expansion in all_expansions}
    if args.mode == "TrainOnly":
        examples = get_examples_from_train(args.train_file, all_expansions)
    if args.mode == "RandomIndex":
        examples = get_examples_from_index(all_expansions, args.para_only)
    expansion_to_sents["examples"] = examples

    fname = "paras" if args.para_only else "sents"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    with open(f"sciad_data/expansions_to_{fname}_{args.mode}_{timestamp}.json", "w") as f:
        json.dump(expansion_to_sents, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Search for examples for each expansion")
    parser.add_argument("--para_only", action="store_true", help="Only search in paragraphs")
    parser.add_argument("--train_file", type=str, default="sciad_data/train.csv", help="Train file")
    parser.add_argument("--mode", type=str, help="Example Selection Mode")
    args = parser.parse_args()
    main(args)
