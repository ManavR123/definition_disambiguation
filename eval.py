import argparse
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from modeling.sgc import get_sgc_embedding
from modeling.utils_modeling import get_baseline_embedding


def get_prediction(expansion_embeddings, acronym, target, acronym_to_expansion):
    preds = {}
    for expansion in acronym_to_expansion[acronym]:
        embed = expansion_embeddings[expansion]
        score = embed @ target / np.linalg.norm(target)
        if type(score) == np.ndarray:
            score = np.median(score)
        preds[expansion] = score

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
    assert args.mode in ["SGC", "Baseline"], f"Mode must be either SGC or Baseline\nGot: {args.mode}"
    expansion_embeddings = np.load(args.expansion_embeddings_path, allow_pickle=True)[()]
    with open("sciad_data/diction.json") as f:
        acronym_to_expansion = json.load(f)
    df = pd.read_csv(filename).dropna(subset=["paper_data"])

    model = AutoModel.from_pretrained(args.model).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

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
        if args.mode == "SGC":
            target, G = get_sgc_embedding(
                model, tokenizer, args.device, acronym, paper_data, text, args.k, args.levels, args.max_examples
            )
        elif args.mode == "Baseline":
            target = get_baseline_embedding(model, tokenizer, args.device, text)

        preds, best = get_prediction(expansion_embeddings, acronym, target, acronym_to_expansion)
        prediction_by_acronym[acronym].append(best)
        gold_by_acronym[acronym].append(gold_expansion)
        if best == gold_expansion:
            correct += 1
        else:
            record_error(
                args.mode, logfile, gold_expansion, paper_id, text, len(G) if args.mode == "SGC" else None, preds, best
            )

    record_results(logfile, correct / len(df), prediction_by_acronym, gold_by_acronym)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the performance of the algorithm.")
    parser.add_argument("-f", "--file", help="The dataset to evaluate on.", required=True)
    parser.add_argument("-k", "--k", help="The number of convolutions to use.", default=4, type=int)
    parser.add_argument("-l", "--levels", help="The number of levels to expand to.", default=2, type=int)
    parser.add_argument("-d", "--device", help="The device to run on.", default="cuda:1")
    parser.add_argument("-m", "--model", help="The model to use.", default="allenai/specter")
    parser.add_argument("--mode", help="The mode to use.", default="SGC")
    parser.add_argument("--max_examples", type=int, help="The maximum number of examples to use.", default=100)
    parser.add_argument(
        "--expansion_embeddings_path",
        help="The expansion embeddings to use.",
        default="sciad_data/expansion_embeddings_normalized_specter_cls.npy",
    )
    args = parser.parse_args()

    logfile = f"logs/{time.strftime('%Y%m%d-%H%M%S')}.txt"
    with open(logfile, "w") as f:
        mode_text = f"SGC with k = {args.k}, levels = {args.levels}" if args.mode == "SGC" else "Baseline"
        print(f"Evaluating on {args.file}", file=f)
        print(f"Mode: {mode_text}", file=f)
        print(f"Expansion Embeddings: {args.expansion_embeddings_path}", file=f)
        print(f"Max Examples: {args.max_examples}", file=f)
        print(f"Model: {args.model}", file=f)

    eval(args.file, args, logfile)
