import argparse
import random
from itsdangerous import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import warnings

warnings.filterwarnings("ignore")
random.seed(42)


def main(args):
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").to("cuda:1")

    expansions = json.load(open(args.expansions, "r"))
    df = pd.read_csv(args.summaries, sep="\t")
    mins, maxes, means, meds, all_dists = [], [], [], [], []
    for senses in tqdm(expansions.values()):
        summaries = [df.loc[df.term == sense, "summary"].values[0] for sense in senses]
        inputs = tokenizer(summaries, return_tensors="pt", truncation=True, padding=True).to("cuda:1")
        with torch.no_grad():
            result = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"]
            mask = mask.unsqueeze(-1).expand(result.size()).float()
            embeddings = torch.sum(result * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

        distances = pairwise_distances(embeddings.cpu().numpy(), metric="euclidean")

        mask = np.triu(np.ones(distances.shape, dtype=bool), k=1)
        mins.append(np.min(distances[mask]))
        maxes.append(np.max(distances[mask]))
        means.append(np.mean(distances[mask]))
        meds.append(np.median(distances[mask]))
        all_dists.extend(distances[mask])

    data = [mins, maxes, means, meds, all_dists]
    plt.title(f"{args.title} (n={len(expansions)})")
    plt.ylabel("Euclidean distance")
    plt.boxplot(data, labels=["Min", "Max", "Mean", "Median", "All"], showfliers=False)
    plt.axis([None, None, 0, 4])
    plt.savefig(args.output)
    print("Median Minimum Distance:", np.median(mins))
    print("Median Maximum Distance:", np.median(maxes))
    print("Median Mean Distance:", np.median(means))
    print("Median Median Distance:", np.median(meds))
    print("Median All Distance:", np.median(all_dists))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expansions", type=str, default="wikipedia_parsing/ambiguous_term_to_senses.json")
    parser.add_argument("--summaries", type=str, default="wikipedia_parsing/ambiguous_terms.tsv")
    parser.add_argument("--output", type=str, default="sense_distribution/ambiguous_boxplot.png")
    parser.add_argument("--title", type=str, default="Distances between senses of ambiguous terms")
    args = parser.parse_args()
    main(args)
