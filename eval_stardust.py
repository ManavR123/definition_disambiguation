import argparse
import json
import time

import pandas as pd
import torch
from tqdm import tqdm

import wandb
from data.dataset import WSDDataset
from modeling.stardust import Stardust
from scorer import record_error, record_results


def main(args):
    wandb.init(project=f"{args.project}_eval")
    wandb.config.update(args)

    with open(args.word_to_senses) as f:
        word_to_senses = json.load(f)
    sense_to_gloss = pd.read_csv(args.sense_dictionary, sep="\t")
    test_data = pd.read_csv(args.file).sample(frac=1.0, random_state=42)
    dataset = WSDDataset(test_data, word_to_senses, sense_to_gloss)

    model = Stardust(
        args.model_name,
        hidden_size=128,
        tie_context_gloss_encoder=True,
        freeze_context_encoder=args.freeze_context_encoder,
        freeze_gloss_encoder=args.freeze_gloss_encoder,
        freeze_paper_encoder=args.freeze_paper_encoder,
    )
    model.load_state_dict(torch.load(args.model_ckpt))
    model.eval().to(args.device)

    logfile = f"logs/{time.strftime('%Y%m%d-%H%M%S')}.txt"
    with open(logfile, "w") as f:
        print(f"Evaluating on {args.file}", file=f)
        print(f"Model: {args.model_ckpt}", file=f)

    preds, golds = [], []
    for batch in tqdm(dataset, total=len(dataset)):
        encoded_contexts, encoded_glosses, encoded_papers = model.create_input(
            [batch["text"]], batch["glosses"], batch["paper_titles"], args.device
        )
        with torch.no_grad():
            context_embeddings, gloss_embeddings, paper_embeddings = model(
                encoded_contexts, encoded_glosses, encoded_papers
            )
            scores = model.get_scores(
                context_embeddings,
                gloss_embeddings,
                paper_embeddings,
                encoded_contexts,
                encoded_glosses,
                [batch["text"]],
                [batch["acronym"]],
                [batch["glosses"]],
                args.device,
            )
        scores = scores.squeeze().cpu().numpy()

        idx = scores.argmax()
        pred = word_to_senses[batch["acronym"]][idx]
        preds.append(pred)
        golds.append(batch["expansion"])

        pred_scores = [(exp, scores[i]) for i, exp in enumerate(word_to_senses[batch["acronym"]])]
        pred_scores = sorted(pred_scores, key=lambda x: x[1], reverse=True)
        if pred != batch["expansion"]:
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
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--freeze_context_encoder", action="store_true")
    parser.add_argument("--freeze_gloss_encoder", action="store_true")
    parser.add_argument("--freeze_paper_encoder", action="store_true")
    args = parser.parse_args()
    main(args)
