import argparse
import itertools
import json
from typing import Dict, List

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb

from modeling.bem import BEM, create_input, get_scores


class WSDDataset(Dataset):
    def __init__(self, train_data: pd.DataFrame, word_to_senses: Dict[str, str], sense_to_gloss: pd.DataFrame):
        data = train_data
        word_to_senses = word_to_senses
        sense_to_gloss = {row["term"]: row["summary"] for _, row in sense_to_gloss.iterrows()}

        self.batches = []
        for _, row in tqdm(data.iterrows(), total=len(data)):
            text = row["text"]
            acronym = row["acronym"]
            labels = torch.Tensor([sense == row["expansion"] for sense in word_to_senses[acronym]]).float()
            glosses = [sense_to_gloss[sense] for sense in word_to_senses[acronym]]
            self.batches.append({"text": text, "acronym": acronym, "labels": labels, "glosses": glosses})

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx: int):
        return self.batches[idx]

    @staticmethod
    def collate_fn(batch: Dict[str, List[any]]):
        return {
            "text": [item["text"] for item in batch],
            "acronym": [item["acronym"] for item in batch],
            "labels": torch.cat([item["labels"] for item in batch], dim=0),
            "glosses": [item["glosses"] for item in batch],
        }


def main(args):
    wandb.init(project=f"{args.project}_train")
    wandb.config.update(args)
    filename = f"models/BEM_{wandb.run.name}.pt"

    with open(args.word_to_senses) as f:
        word_to_senses = json.load(f)
    sense_to_gloss = pd.read_csv(args.sense_dictionary, sep="\t")
    train = pd.read_csv(args.file).sample(frac=1.0, random_state=42)
    dataset = WSDDataset(train, word_to_senses, sense_to_gloss)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn, prefetch_factor=4, num_workers=8
    )

    bem_model = BEM(args.model_name).train().to(args.device)
    optim = torch.optim.Adam(bem_model.parameters(), lr=args.lr)
    wandb.watch(bem_model, log="all", log_freq=args.log_every)

    step, batch_loss = 0, torch.tensor(0.0).to(args.device)
    for _ in tqdm(range(args.num_epochs)):
        for batch in tqdm(dataloader, total=len(dataloader)):
            optim.zero_grad()

            encoded_contexts, encoded_glosses = create_input(
                bem_model.tokenizer, batch["text"], list(itertools.chain(*batch["glosses"])), args.device
            )
            context_embeddings, gloss_embeddings = bem_model(encoded_contexts, encoded_glosses)
            scores = get_scores(
                encoded_contexts,
                context_embeddings,
                gloss_embeddings,
                batch["text"],
                batch["acronym"],
                batch["glosses"],
                args.device,
            )
            loss = F.binary_cross_entropy_with_logits(scores, batch["labels"].to(args.device))
            loss.backward()
            batch_loss += loss.detach()
            optim.step()

            step += 1
            if step % args.log_every == 0:
                wandb.log({"loss": batch_loss.item() / args.log_every})
                batch_loss = torch.tensor(0.0).to(args.device)

        torch.save(bem_model.state_dict(), filename)
        wandb.save(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="The dataset to evaluate on.", default="pseudowords/pseudowords_train.csv")
    parser.add_argument(
        "--word_to_senses",
        type=str,
        help="The path to the word to senses mapping.",
        default="pseudowords/pseudoword_kNN_0.95_expansions.json",
    )
    parser.add_argument("--sense_dictionary", help="The dictionary to use.", default="wikipedia_parsing/terms.tsv")
    parser.add_argument("--project", type=str, help="The project to use.", default="pseudowords")
    parser.add_argument("--device", help="The device to run on.", default="cuda:1")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    args = parser.parse_args()
    main(args)
