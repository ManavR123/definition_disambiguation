import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import wandb

from modeling.baseline import get_average_embedding, get_paper_average_embedding
from modeling.scoring_models import LinearScoring, MLPScoring
from modeling.sent_only_sgc import get_sent_sgc_embedding
from modeling.sgc import get_sgc_embedding
from utils_args import create_parser


def setup_train(args):
    wandb.init(project="acronym_disambiguation_train")
    wandb.config.update(args)

    expansion_embeddings = np.load(args.expansion_embeddings_path, allow_pickle=True)[()]
    wandb.config.update({f"expansion-embedding-{k}": v for k, v in expansion_embeddings.items() if "arg" in k})
    expansion_embeddings = expansion_embeddings["expansion_embeddings"]
    with open("sciad_data/diction.json") as f:
        acronym_to_expansion = json.load(f)
    df = pd.read_csv(args.file).dropna(subset=["paper_data"])

    filename = f"models/{args.scoring_model}_{wandb.run.name}.pt"

    model = AutoModel.from_pretrained(args.model).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.scoring_model == "LinearScoring":
        scoring_model = LinearScoring(model.config.hidden_size, model.config.hidden_size).to(args.device).train()
    elif args.scoring_model == "MLPScoring" and args.graph_mode != "PaperAverage":
        scoring_model = MLPScoring(model.config.hidden_size * 2).to(args.device).train()
    elif args.scoring_model == "MLPScoring" and args.graph_mode == "PaperAverage":
        scoring_model = MLPScoring(model.config.hidden_size * 3).to(args.device).train()

    if args.saved_scoring_model:
        scoring_model.load_state_dict(torch.load(args.saved_scoring_model))
    optim = torch.optim.Adam(scoring_model.parameters(), lr=args.lr)
    return expansion_embeddings, acronym_to_expansion, df, filename, model, tokenizer, scoring_model, optim


def get_target(args, model, tokenizer, acronym, text, paper_data):
    if args.graph_mode == "SGC":
        target, _ = get_sgc_embedding(
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
    if args.graph_mode == "SentSGC":
        target, _ = get_sent_sgc_embedding(
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
        target, _ = get_average_embedding(
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
    if args.graph_mode == "PaperAverage":
        target, _ = get_paper_average_embedding(
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
    return target


def get_loss(expansion_embeddings, scoring_model, acronym, target, gold_expansion, acronym_to_expansion, device):
    target = target.unsqueeze(0).to(device)
    loss = torch.tensor(0.0).to(device)
    for expansion in acronym_to_expansion[acronym]:
        embed = torch.Tensor(expansion_embeddings[expansion]).unsqueeze(0).to(device)
        score = scoring_model(target, embed).squeeze(0).squeeze(1)
        label = 1.0 if expansion == gold_expansion else 0.0
        label = torch.tensor(label).repeat(score.shape[0]).to(device)
        loss += F.binary_cross_entropy_with_logits(score, label)
    return loss


def train(args):
    expansion_embeddings, acronym_to_expansion, df, filename, model, tokenizer, scoring_model, optim = setup_train(args)
    wandb.watch(scoring_model, log="all", log_freq=args.log_every)
    step, batch_loss = 1, 0.0
    for _ in range(args.num_epochs):
        for _, row in tqdm(df.iterrows(), total=len(df)):
            acronym, gold_expansion, text = row["acronym"], row["expansion"], row["text"]
            paper_data = json.loads(row["paper_data"])

            target = get_target(args, model, tokenizer, acronym, text, paper_data)
            loss = get_loss(
                expansion_embeddings, scoring_model, acronym, target, gold_expansion, acronym_to_expansion, args.device
            )
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            batch_loss += loss.item()

            if step % args.batch_size == 0:
                optim.step()

            if step % args.log_every == 0:
                wandb.log({"loss": batch_loss / args.log_every})
                batch_loss = 0.0
            step += 1

        torch.save(scoring_model.state_dict(), filename)
        wandb.save(filename)


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--num_epochs", type=int, help="The number of epochs to train for.", default=10)
    parser.add_argument("--batch_size", type=int, help="The batch size to use.", default=32)
    parser.add_argument("--log_every", type=int, help="The number of steps to log.", default=100)
    parser.add_argument("--lr", type=float, help="The learning rate to use.", default=1e-3)
    args = parser.parse_args()
    train(args)
