import argparse
import json
import random
import time

import git
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from modeling.utils_modeling import get_embeddings


def create_embeddings(args):
    # set seeds
    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device)
    diction = json.load(open("sciad_data/diction.json", "r"))
    expansion_to_sents = json.load(open(args.expansions_to_sents, "r"))

    expansion_to_acronym = {}
    for acronym in diction:
        for expansion in diction[acronym]:
            expansion_to_acronym[expansion] = acronym

    print("Creating embeddings...")
    expansion_embeddings = {}
    for expansion, sents in tqdm(expansion_to_sents["examples"].items()):
        acronym = expansion_to_acronym[expansion].lower()
        if len(sents) > 100:
            sents = random.sample(sents, 100)
        if args.replace_expansion:
            sents = [s.lower().replace(expansion.lower(), f" {acronym} ").strip() for s in sents]

        embeddings = get_embeddings(model, tokenizer, acronym, sents, args.device, args.mode).cpu()
        if args.normalize:
            embeddings = F.normalize(embeddings, dim=-1)

        if args.no_average:
            expansion_embeddings[expansion] = embeddings
        else:
            expansion_embeddings[expansion] = embeddings.mean(dim=0).unsqueeze(0)

    output = {f"arg-{k}": v for k, v in vars(args).items()}
    output["expansion_embeddings"] = expansion_embeddings
    output["expansion_to_sents"] = expansion_to_sents

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    output["git_commit"] = sha

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"sciad_data/expansion_embeddings_{timestamp}.npy"
    np.save(filename, output)
    print(f"Saved expansion embeddings to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create embeddings for acronym expansions")
    parser.add_argument("--expansions_to_sents", type=str)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--mode", type=str, default="mean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_average", action="store_true")
    parser.add_argument("--replace_expansion", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()
    create_embeddings(args)
