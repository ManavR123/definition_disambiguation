import argparse
import json
import random

import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from modeling.utils_modeling import get_embeddings


def create_embeddings(args):
    # set seeds
    random.seed(args.seed)

    expansion_embeddings = {}
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device)
    with open("sciad_data/diction.json") as f:
        diction = json.load(f)

    expansion_to_acronym = {}
    for acronym in diction:
        for expansion in diction[acronym]:
            expansion_to_acronym[expansion] = acronym

    print("Loading examples sentences...")
    expansion_to_sents = {}
    with open(args.expansions_to_sents, "r") as f:
        expansion_to_sents = json.load(f)
    print("Creating embeddings...")
    for expansion in tqdm(expansion_to_sents):
        acronym = expansion_to_acronym[expansion].lower()
        sents = [expansion.lower()] if args.expansion_only else expansion_to_sents[expansion]
        if len(sents) > 100:
            sents = random.sample(sents, 100)
        if args.replace_expansion:
            sents = [s.lower().replace(expansion.lower(), f" {acronym} ") for s in sents]

        embeddings = get_embeddings(model, tokenizer, acronym, sents, args.device, args.mode).cpu()

        if args.no_average:
            expansion_embeddings[expansion] = embeddings
        else:
            expansion_embeddings[expansion] = np.mean(embeddings, axis=0)

    output = {f"arg-{k}": v for k, v in vars(args).items()}
    output["expansion_embeddings"] = expansion_embeddings
    np.save(f"sciad_data/{args.output}.npy", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create embeddings for acronym expansions")
    parser.add_argument("--output", type=str)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--mode", type=str, default="mean")
    parser.add_argument("--expansion_only", action="store_true")
    parser.add_argument("--no_average", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expansions_to_sents", type=str, default="sciad_data/expansions_to_sents.json")
    parser.add_argument("--replace_expansion", action="store_true")
    args = parser.parse_args()

    create_embeddings(args)
