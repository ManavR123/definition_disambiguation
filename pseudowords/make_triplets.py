import argparse
import json
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


def get_embed_matrix(term_embeds):
    idx_to_term = {}
    embeds = []
    for i, (term, embed) in enumerate(term_embeds.items()):
        idx_to_term[i] = term
        embeds.append(embed)
    embeds = np.array(embeds)
    return idx_to_term, embeds


def get_sorted_neighbors(embeds, adjustment):
    distances = pairwise_distances(embeds, metric="euclidean")
    adjusted_distances = np.abs(distances - adjustment * (np.ones(distances.shape) - np.eye(distances.shape[0])))
    sorted_idx = np.argsort(adjusted_distances)
    return sorted_idx


def get_partition(max_size):
    ks = np.arange(2, 9)
    sizes, total, i = [], 0, 0
    while total < max_size:
        k = min(ks[i], max_size - total)
        sizes.append(k)
        total += k
        i = (i + 1) % len(ks)
    if sizes[-1] == 1:
        sizes.pop()
        sizes[0] += 1
    return sizes


def get_knn_pseudowords(term_embeds, adjustment):
    idx_to_term, embeds = get_embed_matrix(term_embeds)
    sorted_neighbors = get_sorted_neighbors(embeds, adjustment)
    partition = get_partition(len(embeds))

    i, count, selected_terms = 0, 0, set()
    expansions = {}
    for k in tqdm(partition):
        while i in selected_terms:
            i += 1

        neighbors = []
        for j in sorted_neighbors[i]:
            if j not in selected_terms:
                neighbors.append(idx_to_term[j])
                selected_terms.add(j)
            if len(neighbors) == k:
                break

        term = idx_to_term[i]
        assert len(neighbors) == k, f"{term} has only {len(neighbors)} neighbors, expected {k}"
        assert len(selected_terms) >= i, f"We have {len(selected_terms)} terms selected, expected at least {i}"

        new_word = "".join([n[0] for n in neighbors]) + f"-{len(expansions)}"
        expansions[new_word] = neighbors
        count += len(neighbors)

    assert count == len(term_embeds), f"{count} != {len(term_embeds)}"
    return expansions


def main(args):
    term_embeds = np.load("pseudowords/terms_embed.npy", allow_pickle=True)[()]["expansion_embeddings"]

    if args.mode == "kNN":
        expansions = get_knn_pseudowords(term_embeds, args.adjustment)

    # save expansions to json
    with open(f"pseudowords/pseudoword_{args.mode}_{args.adjustment}_expansions.json", "w") as f:
        json.dump(expansions, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="kNN")
    parser.add_argument("--adjustment", type=float, default=2.23)
    args = parser.parse_args()
    main(args)
