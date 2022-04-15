import argparse
import random
import json
import pandas as pd
from sklearn.model_selection import train_test_split


def main(args):
    random.seed(args.seed)
    diction = json.load(open(args.dictionary, "r"))
    terms = pd.read_csv(args.terms, sep="\t")
    dataset = pd.read_csv(args.dataset)

    unseen_terms = set(random.choices(terms["term"], k=int(len(terms) * 0.05)))
    unseen_pseudowords = random.sample(diction.keys(), k=int(0.1 * len(diction)))
    for pseudoword in unseen_pseudowords:
        unseen_terms.update(diction[pseudoword])

    percent = len(unseen_terms) / len(terms) * 100
    print(f"Hidding {percent:.2f}% of terms")

    mask = dataset["expansion"].isin(unseen_terms)
    train = dataset[~mask]
    test_1 = dataset[mask]

    train, test_2 = train_test_split(train, test_size=0.1, random_state=42)
    test = pd.concat([test_1, test_2])

    print(f"Train: {len(train)}, Test: {len(test)}")
    train.to_csv("pseudowords/pseudowords_train.csv", index=False)
    test.to_csv("pseudowords/pseudowords_test.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary", type=str, default="pseudowords/pseudoword_kNN_expansions.json")
    parser.add_argument("--dataset", type=str, default="pseudowords/pseudowords_dataset.csv")
    parser.add_argument("--terms", type=str, default="pseudowords/terms.tsv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
