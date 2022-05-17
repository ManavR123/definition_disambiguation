import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from data.dataset import WSDDataset
from modeling.bem import BEM
from modeling.stardust import Stardust
from modeling.wsd_model import WSDModel
from utils.scorer import record_error, record_results


def initialize_model(args) -> WSDModel:
    if args.model_type == "BEM":
        model = BEM(args.model_name)
    elif args.model_type == "Stardust":
        model = Stardust(
            args.model_name,
            hidden_size=128,
            tie_context_gloss_encoder=True,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    return model


def main(args):
    wandb.init(project=f"{args.project}_eval")
    wandb.config.update(args)

    with open(args.word_to_senses) as f:
        word_to_senses = json.load(f)
    sense_to_gloss = pd.read_csv(args.sense_dictionary, sep="\t")
    test_data = pd.read_csv(args.file).sample(frac=1.0, random_state=42)
    dataset = WSDDataset(
        test_data,
        word_to_senses,
        sense_to_gloss,
        reduce_dict=args.reduce_dict,
        context_enhancement=args.context_enhancement,
        citation_enhancement=args.citation_enhancement,
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn, prefetch_factor=4, num_workers=8)

    model = initialize_model(args)
    model.load_state_dict(torch.load(args.model_ckpt))
    model.eval().to(args.device)

    logfile = f"logs/{time.strftime('%Y%m%d-%H%M%S')}.txt"
    with open(logfile, "w") as f:
        print(f"Evaluating on {args.file}", file=f)
        print(f"Model: {args.model_ckpt}", file=f)

    preds, golds = [], []
    for batch in tqdm(dataloader, total=len(dataset)):
        with torch.no_grad():
            scores = model.step(batch, args.device)
        scores = scores.squeeze().cpu().numpy()

        if len(scores.shape) == 0:
            scores = np.expand_dims(scores, 0)

        idx = scores.argmax()
        pred = dataset.word_to_senses[batch["acronym"][0]][idx]
        preds.append(pred)
        golds.append(batch["expansion"][0])

        pred_scores = [(exp, scores[i]) for i, exp in enumerate(dataset.word_to_senses[batch["acronym"][0]])]
        pred_scores = sorted(pred_scores, key=lambda x: x[1], reverse=True)
        if pred != batch["expansion"][0]:
            batch["text"] = batch["text"][0]
            batch["expansion"] = batch["expansion"][0]
            batch["paper_id"] = batch["paper_id"][0]
            record_error(logfile, batch, pred, pred_scores)

    record_results(logfile, preds, golds)
    wandb.save(logfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", help="The dataset to evaluate on.", default="pseudowords/pseudowords_test_not_new.csv"
    )
    parser.add_argument(
        "--word_to_senses",
        type=str,
        help="The path to the word to senses mapping.",
        default="pseudowords/pseudoword_kNN_0.95_expansions.json",
    )
    parser.add_argument("--sense_dictionary", help="The dictionary to use.", default="wikipedia_parsing/terms.tsv")
    parser.add_argument("--project", type=str, help="The project to use.", default="pseudowords")
    parser.add_argument("--device", help="The device to run on.", default="cuda:1")
    parser.add_argument("--reduce_dict", action="store_true", help="Whether to reduce the dictionary.")
    parser.add_argument("--context_enhancement", action="store_true", help="Whether to use context enhancement.")
    parser.add_argument("--citation_enhancement", action="store_true", help="Whether to use citation enhancement.")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--model_type", type=str, default="BEM", choices=["BEM", "Stardust"])
    parser.add_argument("--model_ckpt", type=str, required=True)
    args = parser.parse_args()
    main(args)
