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
    text = []
    if type(paper_data["abstract"]) == list:
        text.extend(paper_data["abstract"])
    elif type(paper_data["abstract"]) == str:
        text.append(paper_data["abstract"])
    text.extend(paper_data.get("pdf_parse", []))

    examples = []
    for para in text:
        sents = sent_tokenize(para)
        for sent in sents:
            sent = sent.lower().replace("-", " ")
            if term in sent:
                examples.append(sent)
    return examples


def main(args):
    diction = json.load(open(args.dictionary, "r"))
    rows = []
    for pseudoword in tqdm(diction):
        for i, term in enumerate(diction[pseudoword]):
            res = search(term, fields=["abstract", "pdf_parse"], limit=args.num_examples)
            term_examples = []
            for paper in res:
                examples = get_examples(paper, term)
                for j, text in enumerate(examples):
                    extra_text = examples[(j + 1) % len(examples)].replace(term, pseudoword)
                    term_examples.append(
                        {
                            "acronym": pseudoword,
                            "expansion": term,
                            "text": text.replace(term, pseudoword),
                            "examples": [text.replace(term, pseudoword), extra_text],
                            "paper_titles": [paper["title"]],
                            "paper_id": paper["paper_id"],
                        }
                    )

            assert len(term_examples) > 0, f"No examples found for {term}"
            k = round(SENSE_DISTRIBUTIONS[len(diction[pseudoword])][i] * args.num_examples)

            if k > len(term_examples):
                print(f"{pseudoword} {term} only has {len(term_examples)} examples, expected at least {k}")
                k = len(term_examples)

            rows.extend(random.sample(term_examples, k=k))

    print(f"Found {len(rows)} examples")
    df = pd.DataFrame(rows)
    df.to_csv("pseudowords/pseudowords_dataset.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary", type=str, default="pseudowords/pseudoword_kNN_expansions.json")
    parser.add_argument("--num_examples", type=int, default=1000)
    args = parser.parse_args()
    main(args)
