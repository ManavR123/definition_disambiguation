import argparse
import json
import random
import time

from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
import wandb

from indexing.utils_index import search
from modeling.utils_modeling import get_embeddings


def create_embeddings(args):
    # set seeds
    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    wandb.init(project="acronym_disambiguation")
    wandb.config.update(args)
    
    with open("sciad_data/diction.json") as f:
        diction = json.load(f)
    
    all_expansions = []
    for exps in diction.values():
        all_expansions.extend(exps)
    
    print("Collecting examples sentences...")
    expansion_to_sents = {}
    for expansion in tqdm(all_expansions):
        sents = []
        results = search(expansion, ["pdf_parse"], 100)
        for result in results:
            for para in result.get("pdf_parse", []):
                para_sents = sent_tokenize(para)
                for sent in para_sents:
                    if expansion in sent.lower():
                        sents.append(para)
        expansion_to_sents[expansion] = sents

    step = 1
    for _ in range(args.num_epochs):
        for expansions in tqdm(diction.values()):
            expansion_embeddings = torch.zeros(len(expansions), model.config.hidden_size)
            for expansion in expansions:
                sents = expansion_to_sents[expansion]
                if len(sents) > 100:
                    sents = random.sample(sents, 100)

                embeddings = get_embeddings(model, tokenizer, sents, args.device, args.mode, is_train=True)
                expansion_embeddings[expansions.index(expansion)] = embeddings.mean(dim=0)

            loss = torch.sum(expansion_embeddings @ expansion_embeddings.T - torch.eye(expansion_embeddings.shape[0]))
            if step % args.log_every == 0:
                wandb.log({"loss": loss.item()})
            optim.zero_grad()
            loss.backward()
            optim.step()
            step += 1
    
    filename = time.strftime('%Y%m%d-%H%M%S')
    model.save_pretrained(f"models/embedding_model_{filename}")
    wandb.save(f"models/embedding_model_{filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create embeddings for acronym expansions")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--mode", type=str, default="mean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=1)
    args = parser.parse_args()

    create_embeddings(args)
