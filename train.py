from fileinput import filename
import json
from isort import file
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from modeling.scoring_models import LinearScoring
import wandb
import time
from utils_args import create_parser
from modeling.sgc import get_sgc_embedding


def get_loss(expansion_embeddings, scoring_model, acronym, target, gold_expansion, acronym_to_expansion, device):
    target = scoring_model(torch.Tensor(target / np.linalg.norm(target)).to(device))
    target = target / torch.norm(target)
    loss = torch.tensor(0.0).to(device)
    for expansion in acronym_to_expansion[acronym]:
        embed = torch.Tensor(expansion_embeddings[expansion]).to(device)
        score = embed @ target
        if len(score.shape) >= 1:
            score = torch.median(score)
        label = 1.0 if expansion == gold_expansion else 0.0
        loss += F.binary_cross_entropy_with_logits(score, torch.tensor(label).to(device))
    return loss


def train(args):
    expansion_embeddings = np.load(args.expansion_embeddings_path, allow_pickle=True)[()]
    with open("sciad_data/diction.json") as f:
        acronym_to_expansion = json.load(f)
    df = pd.read_csv(args.file).dropna(subset=["paper_data"])

    wandb.init(project="acronym_disambiguation")
    wandb.config.update(args)

    model = AutoModel.from_pretrained(args.model).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    scoring_model = LinearScoring(model.config.hidden_size, model.config.hidden_size).to(args.device)
    optim = torch.optim.Adam(scoring_model.parameters())

    wandb.watch(scoring_model)
    step = 1
    batch_loss = 0.0
    for _ in range(args.num_epochs):
        for _, row in tqdm(df.iterrows(), total=len(df)):
            acronym = row["acronym"]
            gold_expansion = row["expansion"]
            paper_data = json.loads(row["paper_data"])
            text = row["text"]

            target, _ = get_sgc_embedding(
                model, tokenizer, args.device, acronym, paper_data, text, args.k, args.levels, args.max_examples
            )

            loss = get_loss(
                expansion_embeddings, scoring_model, acronym, target, gold_expansion, acronym_to_expansion, args.device
            )
            optim.zero_grad()
            loss.backward()

            batch_loss += loss.item()

            if step % args.batch_size == 0:
                optim.step()

            if step % args.log_every == 0:
                wandb.log({"loss": batch_loss / args.log_every})
                batch_loss = 0.0

            step += 1

    filename = f"models/scoring_model_{time.strftime('%Y%m%d-%H%M%S')}.pt"
    torch.save(scoring_model.state_dict(), filename)
    wandb.save(filename)


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--num_epochs", type=int, help="The number of epochs to train for.", default=10)
    parser.add_argument("--batch_size", type=int, help="The batch size to use.", default=32)
    parser.add_argument("--log_every", type=int, help="The number of steps to log.", default=100)
    args = parser.parse_args()
    train(args)
