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
    model = AutoModel.from_pretrained(args.model).to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    wandb.init(project="acronym_disambiguation")
    wandb.config.update(args)

    with open("sciad_data/diction.json") as f:
        diction = json.load(f)

    print("Loading examples sentences...")
    expansion_to_sents = {}
    with open(args.expansions_to_sents, "r") as f:
        expansion_to_sents = json.load(f)

    print("Beginning training...")
    step = 0
    total_loss = 0
    for _ in range(args.num_epochs):
        for expansions in tqdm(diction.values()):
            if len(expansions) > 3:
                expansions = random.sample(expansions, 3)
            expansion_embeddings = torch.zeros(len(expansions), model.config.hidden_size)
            loss = 0
            for expansion in expansions:
                sents = expansion_to_sents[expansion]
                if len(sents) == 0:
                    print(f"No sentences found for expansion {expansion}")
                    continue
                if len(sents) > args.batch_size:
                    sents = random.sample(sents, args.batch_size)

                embeddings = get_embeddings(model, tokenizer, sents, args.device, args.mode, is_train=True)
                embeddings = F.normalize(embeddings, dim=1)
                loss -= torch.sum(embeddings @ expansion_embeddings.T) - len(sents)
                expansion_embeddings[expansions.index(expansion)] = F.normalize(embeddings.mean(dim=0))

            loss += torch.sum(expansion_embeddings @ expansion_embeddings.T - torch.eye(expansion_embeddings.shape[0]))
            loss.backward()
            step += 1

            if step % args.optim_every == 0:
                optim.step()
                optim.zero_grad()

            total_loss += loss.item()
            if step % args.log_every == 0:
                wandb.log({"loss": total_loss / args.log_every})
                total_loss = 0

    filename = time.strftime("%Y%m%d-%H%M%S")
    model.save_pretrained(f"models/embedding_model_{filename}")
    tokenizer.save_pretrained(f"models/embedding_model_{filename}")
    wandb.save(f"models/embedding_model_{filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create embeddings for acronym expansions")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--mode", type=str, default="mean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--expansions_to_sents", type=str, default="sciad_data/expansions_to_sents.json")
    parser.add_argument("--optim_every", type=int, default=8)
    args = parser.parse_args()

    train_embeddings(args)
