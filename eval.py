import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from modeling.scoring_models import IdentityScoring, LinearScoring
from modeling.sgc import get_sgc_embedding
from modeling.baseline import get_average_embedding, get_baseline_embedding
from utils_args import create_parser


def get_prediction(expansion_embeddings, scoring_model, acronym, target, acronym_to_expansion, device):
    with torch.no_grad():
        target = scoring_model(torch.Tensor(target / np.linalg.norm(target)).to(device))
    target = target / torch.norm(target)

    preds = {}
    for expansion in acronym_to_expansion[acronym]:
        embed = torch.Tensor(expansion_embeddings[expansion]).to(device)
        score = embed @ target
        if len(score.shape) >= 1:
            score = torch.median(score)
        preds[expansion] = score.cpu().item()

    best = max(preds, key=preds.get)
    return preds, best


def record_error(mode, logfile, gold_expansion, paper_id, text, graph_size, preds, best):
    with open(logfile, "a") as f:
        print("==========================================================", file=f)
        print(f"Text: {text}", file=f)
        print(f"Prediction: {best}\nGold: {gold_expansion}", file=f)
        print(f"Scores: {preds}", file=f)
        print(f"Paper ID: {paper_id}", file=f)
        if mode == "SGC":
            print(f"Graph Size: {graph_size}", file=f)


def record_results(logfile, accuracy, prediction_by_acronym, gold_by_acronym):
    with open(logfile, "a") as f:
        print("**********************************************************", file=f)
        print(f"Accuracy: {accuracy}", file=f)
        for name, score_func in [("F1", f1_score), ("Precision", precision_score), ("Recall", recall_score)]:
            score = np.mean(
                [
                    score_func(
                        gold_by_acronym[acronym], prediction_by_acronym[acronym], average="macro", zero_division=0
                    )
                    for acronym in prediction_by_acronym
                ]
            )
            print(f"{name}: {score}", file=f)


def eval(filename, args, logfile):
    assert args.graph_mode in [
        "SGC",
        "Average",
        "Baseline",
    ], f"Mode must be either SGC, Average or Baseline\nGot: {args.graph_mode}"
    expansion_embeddings = np.load(args.expansion_embeddings_path, allow_pickle=True)[()]
    with open("sciad_data/diction.json") as f:
        acronym_to_expansion = json.load(f)
    df = pd.read_csv(filename).dropna(subset=["paper_data"])

    model = AutoModel.from_pretrained(args.model).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.scoring_model:
        scoring_model = LinearScoring(model.config.hidden_size, model.config.hidden_size).to(args.device)
        scoring_model.load_state_dict(torch.load(args.scoring_model))
    else:
        scoring_model = IdentityScoring().to(args.device)

    correct = 0
    prediction_by_acronym = defaultdict(list)
    gold_by_acronym = defaultdict(list)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        acronym = row["acronym"]
        gold_expansion = row["expansion"]
        paper_data = json.loads(row["paper_data"])
        paper_id = paper_data["paper_id"]
        text = row["text"]

        target = None
        if args.graph_mode == "SGC":
            target, G = get_sgc_embedding(
                model,
                tokenizer,
                args.device,
                acronym,
                paper_data,
                text,
                args.k,
                args.levels,
                args.max_examples,
                args.embedding_mode,
            )
        if args.graph_mode == "Average":
            target, G = get_average_embedding(
                model,
                tokenizer,
                args.device,
                acronym,
                paper_data,
                text,
                args.levels,
                args.max_examples,
                args.embedding_mode,
            )
        elif args.graph_mode == "Baseline":
            target = get_baseline_embedding(model, tokenizer, args.device, text, args.embedding_mode)

        preds, best = get_prediction(
            expansion_embeddings, scoring_model, acronym, target, acronym_to_expansion, args.device
        )
        prediction_by_acronym[acronym].append(best)
        gold_by_acronym[acronym].append(gold_expansion)
        if best == gold_expansion:
            correct += 1
        else:
            record_error(
                args.graph_mode,
                logfile,
                gold_expansion,
                paper_id,
                text,
                len(G) if args.graph_mode == "SGC" else None,
                preds,
                best,
            )

    record_results(logfile, correct / len(df), prediction_by_acronym, gold_by_acronym)


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--scoring_model", default=None, type=str, help="Path to scoring model")
    args = parser.parse_args()

    logfile = f"logs/{time.strftime('%Y%m%d-%H%M%S')}.txt"
    with open(logfile, "w") as f:
        mode_text = (
            f"{args.graph_mode} with k = {args.k}, levels = {args.levels}"
            if args.graph_mode != "Baseline"
            else "Baseline"
        )
        print(f"Evaluating on {args.file}", file=f)
        print(f"Mode: {mode_text}", file=f)
        print(f"Expansion Embeddings: {args.expansion_embeddings_path}", file=f)
        print(f"Max Examples: {args.max_examples}", file=f)
        print(f"Model: {args.model}", file=f)
        print(f"Scoring Model: {args.scoring_model if args.scoring_model else 'Identity'}", file=f)
        print(f"Embedding Mode: {args.embedding_mode}", file=f)

    eval(args.file, args, logfile)
