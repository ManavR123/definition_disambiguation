import argparse
import json
import random
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import wandb
from modeling.utils_modeling import get_embeddings


def train_embeddings(args):
    # set seeds
    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device).train()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    wandb.init(project="acronym_disambiguation_contrastive_train")
    wandb.config.update(args)

    diction = json.load(open("sciad_data/diction.json"))
    expansion_to_sents = json.load(open(args.expansions_to_sents, "r"))

    print("Beginning training...")
    step = 0
    total_loss = 0
    for _ in range(args.num_epochs):
        for acronym, expansions in tqdm(diction.items()):
            if len(expansions) > 3:
                expansions = random.sample(expansions, 3)
            expansion_embeddings = torch.zeros(len(expansions), model.config.hidden_size)
            loss = 0
            for expansion in expansions:
                sents = expansion_to_sents[expansion]
                if len(sents) > args.batch_size:
                    sents = random.sample(sents, args.batch_size)
                if args.replace_expansion:
                    sents = [s.lower().replace(expansion.lower(), f" {acronym} ") for s in sents]

                embeddings = get_embeddings(
                    model,
                    tokenizer,
                    acronym,
                    sents,
                    args.device,
                    args.mode,
                    is_train=True,
                )
                scores = embeddings @ embeddings.T
                loss -= torch.sum(scores - torch.diag(scores))
                expansion_embeddings[expansions.index(expansion)] = embeddings.mean(dim=0)

            scores = expansion_embeddings @ expansion_embeddings.T
            loss += torch.sum(scores - torch.diag(scores))
            loss /= len(expansions)
            loss.backward()
            total_loss += loss.item()
            step += 1

            if step % args.optim_every == 0:
                optim.step()
                optim.zero_grad()

            if step % args.log_every == 0:
                wandb.log({"loss": total_loss / args.log_every})
                total_loss = 0

    filename = wandb.run.name
    model.save_pretrained(f"models/embedding_model_{filename}")
    tokenizer.save_pretrained(f"models/embedding_model_{filename}")
    wandb.save(f"models/embedding_model_{filename}/*")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create embeddings for acronym expansions")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--mode", type=str, default="acronym")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--expansions_to_sents", type=str, default="sciad_data/expansions_to_sents.json")
    parser.add_argument("--optim_every", type=int, default=8)
    parser.add_argument("--replace_expansion", action="store_true")
    args = parser.parse_args()

    train_embeddings(args)
