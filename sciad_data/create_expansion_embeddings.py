import argparse
import json
import time

import git
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from modeling.utils_modeling import get_embeddings


def create_embeddings(args):
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device)
    diction = json.load(open(args.dictionary "r"))

    expansions = []
    for exps in diction.values():
        expansions.extend(exps)

    print("Creating embeddings...")
    expansion_embeddings = {}
    for expansion in tqdm(expansion):
        embeddings = get_embeddings(model, tokenizer, [], [expansion], args.device, args.mode).cpu()
        if args.normalize:
            embeddings = F.normalize(embeddings, dim=-1)
        expansion_embeddings[expansion] = embeddings

    output = {f"arg-{k}": v for k, v in vars(args).items()}
    output["expansion_embeddings"] = expansion_embeddings
    output["git_commit"] = sha

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"sciad_data/expansion_embeddings_{timestamp}.npy"
    np.save(filename, output)
    print(f"Saved expansion embeddings to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create embeddings for acronym expansions")
    parser.add_argument("--dictionary", type=str, default="sciad_data/diction.json", help="Path to dictionary")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--mode", type=str, default="mean")
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()
    create_embeddings(args)
