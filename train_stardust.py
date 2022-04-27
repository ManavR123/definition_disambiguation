import argparse
import itertools
import json

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
import wandb

from data.dataset import WSDDataset
from modeling.stardust import Stardust


def main(args):
    wandb.init(project=f"{args.project}_train")
    wandb.config.update(args)
    filename = f"models/Stardust_{wandb.run.name}.pt"

    with open(args.word_to_senses) as f:
        word_to_senses = json.load(f)
    sense_to_gloss = pd.read_csv(args.sense_dictionary, sep="\t")
    train = pd.read_csv(args.file).sample(frac=1.0, random_state=42)
    dataset = WSDDataset(train, word_to_senses, sense_to_gloss, is_train=True)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn, prefetch_factor=4, num_workers=8
    )

    model = Stardust(
        args.model_name,
        hidden_size=128,
        tie_context_gloss_encoder=True,
        freeze_context_encoder=args.freeze_context_encoder,
        freeze_gloss_encoder=args.freeze_gloss_encoder,
        freeze_paper_encoder=args.freeze_paper_encoder,
    )
    model.train().to(args.device)
    optim = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=10000, num_training_steps=len(dataloader) * args.num_epochs
    )
    wandb.watch(model, log="all", log_freq=args.log_every)

    step, batch_loss = 0, torch.tensor(0.0).to(args.device)
    for _ in tqdm(range(args.num_epochs)):
        for batch in tqdm(dataloader, total=len(dataloader)):
            optim.zero_grad()

            encoded_contexts, encoded_glosses, encoded_papers = model.create_input(
                batch["text"], list(itertools.chain(*batch["glosses"])), batch["paper_titles"], args.device
            )
            context_embeddings, gloss_embeddings, paper_embeddings = model(
                encoded_contexts, encoded_glosses, encoded_papers
            )
            scores = model.get_scores(
                context_embeddings,
                gloss_embeddings,
                paper_embeddings,
                encoded_contexts,
                encoded_glosses,
                batch["text"],
                batch["acronym"],
                batch["glosses"],
                args.device,
            )
            loss = F.binary_cross_entropy_with_logits(scores, batch["labels"].to(args.device))
            loss.backward()
            batch_loss += loss.detach()
            optim.step()
            scheduler.step()

            step += 1
            if step % args.log_every == 0:
                wandb.log({"loss": batch_loss.item() / args.log_every})
                batch_loss = torch.tensor(0.0).to(args.device)

        torch.save(model.state_dict(), filename)
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
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--freeze_context_encoder", action="store_true")
    parser.add_argument("--freeze_gloss_encoder", action="store_true")
    parser.add_argument("--freeze_paper_encoder", action="store_true")
    args = parser.parse_args()
    main(args)
