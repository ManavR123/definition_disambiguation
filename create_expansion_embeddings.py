import argparse
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from elasticsearch import Elasticsearch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

es = Elasticsearch(hosts=["http://localhost:9200"], timeout=60, retry_on_timeout=True)


def search(text_query, limit=100):
    query = {
        "multi_match": {
            "query": f'"{text_query}"',
            "fields": ["pdf_parse"],
            "type": "phrase",
        }
    }

    return [result["_source"] for result in es.search(index="s2orc", size=limit, query=query)["hits"]["hits"]]


def get_embeddings(model, tokenizer, sents, mode):
    inputs = tokenizer(
        sents,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    ).to(args.device)
    with torch.no_grad():
        result = model(**inputs)
    if mode == "CLS":
        embeds = result.last_hidden_state[:, 0, :]
    elif mode == "mean":
        token_embeddings = result[0]
        input_mask_expanded = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        embeds = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    embeds = F.normalize(embeds, p=2, dim=1).detach().cpu().numpy()
    return embeds


def create_embeddings(args):
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

        results = [] if args.expansion_only else search(expansion, 100)
        for result in results:
            for para in result.get("pdf_parse", []):
                if expansion in para.lower():
                    sents.append(para)

        if len(sents) == 0:
            print(f"{expansion} not found")
            continue

        if len(sents) > 100:
            sents = random.sample(sents, 100)

        embeddings = get_embeddings(model, tokenizer, sents, args.mode)

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
    args = parser.parse_args()

    create_embeddings(args)
