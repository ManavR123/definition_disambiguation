import json
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import wandb
from modeling.baseline import get_average_embedding, get_paper_average_embedding
from modeling.scoring_models import IdentityScoring, LinearScoring, MLPScoring
from scorer import score_expansion
from utils_args import create_parser


def get_prediction(expansion_embeddings, scoring_model, acronym, target, acronym_to_expansion, device):
    target = target.unsqueeze(0).to(device)
    preds = {}
    for expansion in acronym_to_expansion[acronym]:
        embed = torch.Tensor(expansion_embeddings[expansion]).unsqueeze(0).to(device)
        with torch.no_grad():
            score = scoring_model(target, embed).squeeze()
        if len(score.shape) >= 1:
            score = torch.median(score)
        preds[expansion] = score.cpu().item()
    best = max(preds, key=preds.get)
    # sort by score
    preds = dict(sorted(preds.items(), key=lambda item: item[1], reverse=True))
    return preds, best


def record_error(mode, logfile, gold_expansion, paper_id, text, graph_size, preds, best):
    with open(logfile, "a") as f:
        print("==========================================================", file=f)
        print(f"Text: {text}", file=f)
        print(f"Prediction: {best}\nGold: {gold_expansion}", file=f)
        print(f"Scores: {preds}", file=f)
        print(f"Paper ID: {paper_id}", file=f)
        if mode != "Baseline":
            print(f"Graph Size: {graph_size}", file=f)


def record_results(logfile, predictions, golds):
    scores = {}
    with open(logfile, "a") as f:
        print("**********************************************************", file=f)
        acc, prec, rec, f1 = score_expansion(golds, predictions)
        for name, score in [("accuracy", acc), ("F1", f1), ("Precision", prec), ("Recall", rec)]:
            print(f"{name}: {score}", file=f)
            scores[name] = score
    with open(f"predictions/{wandb.run.name}_preds.txt", "w") as f:
        f.write("\n".join(predictions))
    wandb.log(scores)


def get_target(args, model, tokenizer, acronym, examples, paper_titles):
    if args.graph_mode == "Average":
        target, G = get_average_embedding(
            model,
            tokenizer,
            args.device,
            acronym,
            examples,
            args.embedding_mode,
        )
    if args.graph_mode == "PaperAverage":
        target, G = get_paper_average_embedding(
            model,
            tokenizer,
            args.device,
            acronym,
            examples,
            paper_titles,
            args.embedding_mode,
        )
    return target, G


def eval_model(filename, args, logfile):
    assert args.graph_mode in [
        "Average",
        "PaperAverage",
        "Baseline",
    ], f"Mode must be either SGC, SentSGC, Average, PaperAverage or Baseline\nGot: {args.graph_mode}"

    expansion_embeddings = np.load(args.expansion_embeddings_path, allow_pickle=True)[()]
    wandb.config.update({f"expansion-embedding-{k}": v for k, v in expansion_embeddings.items() if "arg" in k})
    expansion_embeddings = expansion_embeddings["expansion_embeddings"]

    with open(args.dictionary) as f:
        acronym_to_expansion = json.load(f)
    df = pd.read_csv(filename)

    model = AutoModel.from_pretrained(args.model).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.scoring_model == "LinearScoring":
        scoring_model = LinearScoring(model.config.hidden_size, model.config.hidden_size).to(args.device)
        scoring_model.load_state_dict(torch.load(args.saved_scoring_model))
    elif args.scoring_model == "MLPScoring" and args.graph_mode != "PaperAverage":
        scoring_model = MLPScoring(model.config.hidden_size * 2).to(args.device)
        scoring_model.load_state_dict(torch.load(args.saved_scoring_model))
    elif args.scoring_model == "MLPScoring" and args.graph_mode == "PaperAverage":
        scoring_model = MLPScoring(model.config.hidden_size * 3).to(args.device)
        scoring_model.load_state_dict(torch.load(args.saved_scoring_model))
    else:
        scoring_model = IdentityScoring().to(args.device)
    scoring_model.eval()

    predictions, golds = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        acronym = row["acronym"]
        gold_expansion = row["expansion"]
        text = row["text"]
        examples = eval(row["examples"])
        paper_titles = eval(row["paper_titles"])
        paper_id = row["paper_id"]

        target, G = get_target(args, model, tokenizer, acronym, examples, paper_titles)
        pred_scores, best = get_prediction(
            expansion_embeddings, scoring_model, acronym, target, acronym_to_expansion, args.device
        )
        predictions.append(best)
        golds.append(gold_expansion)
        if best != gold_expansion:
            record_error(args.graph_mode, logfile, gold_expansion, paper_id, text, len(G), pred_scores, best)

    record_results(logfile, predictions, golds)
    wandb.save(logfile)
    wandb.save(args.expansion_embeddings_path)
    wandb.save(args.saved_scoring_model)


if __name__ == "__main__":
    parser = create_parser()
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
        if args.saved_scoring_model:
            print(f"Saved Scoring Model: {args.saved_scoring_model}", file=f)
        print(f"Embedding Mode: {args.embedding_mode}", file=f)

    wandb.init(project=f"{args.project}_eval")
    wandb.config.update(args)

    eval_model(args.file, args, logfile)
