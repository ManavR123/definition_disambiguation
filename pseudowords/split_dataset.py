import argparse
import json
import random

import pandas as pd
from sklearn.model_selection import train_test_split


def get_most_frequent_senses(df):
    return set(
        df.groupby(["acronym", "expansion"], group_keys=False)
        .agg({"expansion": ["count"]})
        .sort_values(["acronym", ("expansion", "count")], ascending=True)
        .groupby("acronym")
        .tail(1)
        .reset_index()["expansion"][""]
        .tolist()
    )


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

    train_3, test_not_new = train_test_split(train_2, test_size=0.1, random_state=42)
    train, dev = train_test_split(train_3, test_size=0.1, random_state=42)

    mfs = get_most_frequent_senses(dataset)
    mask = test_not_new["expansion"].isin(mfs)
    test_mfs = test_not_new[mask]
    test_lfs = test_not_new[~mask]

    train.to_csv("pseudowords/pseudowords_train.csv", index=False)
    dev.to_csv("pseudowords/pseudowords_dev.csv", index=False)
    test_new_pseudowords.to_csv("pseudowords/pseudowords_test_new_pseudowords.csv", index=False)
    test_new_senses.to_csv("pseudowords/pseudowords_test_new_senses.csv", index=False)
    test_not_new.to_csv("pseudowords/pseudowords_test_not_new.csv", index=False)
    test_mfs.to_csv("pseudowords/pseudowords_test_mfs.csv", index=False)
    test_lfs.to_csv("pseudowords/pseudowords_test_lfs.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary", type=str, default="pseudowords/pseudoword_kNN_0.95_expansions.json")
    parser.add_argument("--dataset", type=str, default="pseudowords/pseudowords_dataset.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
