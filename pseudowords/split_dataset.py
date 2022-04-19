import argparse
import json
import random

import pandas as pd
from sklearn.model_selection import train_test_split


def main(args):
    random.seed(args.seed)
    diction = json.load(open(args.dictionary, "r"))
    dataset = pd.read_csv(args.dataset)

    pseudoword_to_num_senses = [{"pseudoword": k, "num_senses": len(v)} for k, v in diction.items()]
    df = pd.DataFrame(pseudoword_to_num_senses)
    unseen_pseudowords = set(
        df.groupby("num_senses", group_keys=False)
        .apply(lambda x: x.sample(int(0.1 * len(diction) // 7)))["pseudoword"]
        .to_list()
    )

    mask = dataset["acronym"].isin(unseen_pseudowords)
    train_1 = dataset[~mask]
    test_new_pseudowords = dataset[mask]

    seen_pseudowords = set(diction.keys()) - unseen_pseudowords
    senses = []
    for pseudoword in seen_pseudowords:
        senses.extend(diction[pseudoword])
    unseen_senses = set(random.sample(senses, k=int(0.1 * len(senses))))

    mask = train_1["expansion"].isin(unseen_senses)
    train_2 = train_1[~mask]
    test_new_senses = train_1[mask]

    train, test_not_new = train_test_split(train_2, test_size=0.1, random_state=42)

    train.to_csv("pseudowords/pseudowords_train.csv", index=False)
    test_new_pseudowords.to_csv("pseudowords/pseudowords_test_new_pseudowords.csv", index=False)
    test_new_senses.to_csv("pseudowords/pseudowords_test_new_senses.csv", index=False)
    test_not_new.to_csv("pseudowords/pseudowords_test_not_new.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary", type=str, default="pseudowords/pseudoword_kNN_0.95_expansions.json")
    parser.add_argument("--dataset", type=str, default="pseudowords/pseudowords_dataset.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
