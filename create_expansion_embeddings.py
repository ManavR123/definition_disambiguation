import argparse
import json
import random

import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from indexing.utils_index import search
from modeling.utils_modeling import get_embeddings



def create_embeddings(args):
    # set seeds
    random.seed(args.seed)

    expansion_embeddings = {}
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device)

    with open("sciad_data/diction.json") as f:
        diction = json.load(f)

    expansions = []
    for exps in diction.values():
        expansions.extend(exps)

    for expansion in tqdm(expansions):
        sents = [expansion]

        results = [] if args.expansion_only else search(expansion, ["pdf_parse"], 100)
        for result in results:
            for para in result.get("pdf_parse", []):
                if expansion in para.lower():
                    sents.append(para)

        if len(sents) > 100:
            sents = random.sample(sents, 100)

        embeddings = get_embeddings(model, tokenizer, sents, args.device, args.mode).cpu().numpy()

        if args.no_average:
            expansion_embeddings[expansion] = embeddings
        else:
            expansion_embeddings[expansion] = np.mean(embeddings, axis=0)

    np.save(f"sciad_data/{args.output}.npy", expansion_embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create embeddings for acronym expansions")
    parser.add_argument("--output", type=str, default="expansion_embeddings")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--model", type=str, default="allenai/specter")
    parser.add_argument("--mode", type=str, default="CLS")
    parser.add_argument("--expansion_only", action="store_true")
    parser.add_argument("--no_average", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_embeddings(args)
