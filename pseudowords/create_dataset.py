import argparse
import json
import random
import pandas as pd
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from indexing.utils_index import search

SENSE_DISTRIBUTIONS = {
    1: [1.0],
    2: [0.878, 0.122],
    3: [0.781, 0.179, 0.04],
    4: [0.729, 0.196, 0.064, 0.011],
    5: [0.711, 0.188, 0.074, 0.023, 0.004],
    6: [0.654, 0.215, 0.085, 0.031, 0.012, 0.003],
    7: [0.611, 0.230, 0.095, 0.041, 0.018, 0.004, 0.001],
    8: [0.627, 0.209, 0.093, 0.044, 0.018, 0.006, 0.002, 0.001],
}


def get_examples(paper_data, term):
    examples = []
    for para in paper_data.get("pdf_parse", []):
        sents = sent_tokenize(para)
        for sent in sents:
            if f"{term}".lower() in f" {sent} ".lower():
                examples.append(sent.lower())
    return examples


def main(args):
    diction = json.load(open(args.dictionary, "r"))
    rows = []
    for pseudoword in tqdm(diction):
        for i, term in enumerate(diction[pseudoword]):
            res = search(term, fields=["pdf_parse"], limit=30)
            term_examples = []
            for paper in res:
                examples = get_examples(paper, term)
                examples = random.choices(examples, k=min(len(examples), 3))
                for example in examples:
                    term_examples.append(
                        {
                            "acronym": pseudoword.lower(),
                            "expansion": term,
                            "text": example.replace(term.lower(), pseudoword.lower()),
                            "paper_data": paper,
                        }
                    )

            assert len(term_examples) > 0, f"No examples found for {term}"
            k = min(int(SENSE_DISTRIBUTIONS[len(diction[pseudoword])][i] * args.num_examples), len(term_examples))
            rows.extend(random.choices(term_examples, k=k))

    print(f"Found {len(rows)} examples")
    df = pd.DataFrame(rows)
    df.to_csv("pseudowords/pseudowords_dataset.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary", type=str, default="pseudowords/pseudoword_kNN_expansions.json")
    parser.add_argument("--num_examples", type=int, default=1000)
    args = parser.parse_args()
    main(args)
