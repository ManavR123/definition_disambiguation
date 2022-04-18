import argparse
import json

import pandas as pd
from elasticsearch import Elasticsearch
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

es = Elasticsearch(hosts=["http://localhost:9200"], timeout=60, retry_on_timeout=True)


def get_examples(text, acronym, paper_data, max_examples):
    examples = []
    for para in paper_data.get("pdf_parse", []):
        sents = sent_tokenize(para)
        for sent in sents:
            if len(examples) >= max_examples:
                return examples
            if acronym in sent and text not in sent:
                examples.append(sent)
    return examples


def process_paper(text, acronym, paper_data, paper_id, seen_papers, levels):
    seen_papers.add(paper_id)
    if levels <= 0:
        return

    for cite in paper_data["inbound_citations"] + paper_data["outbound_citations"]:
        if cite in seen_papers:
            continue
        cite_data = es.get(index="s2orc", id=cite)["_source"]
        process_paper(text, acronym, cite_data, cite, seen_papers, levels - 1)


def get_paper_text(paper_data):
    abstract = paper_data["abstract"] if paper_data["abstract"] != None else ""
    title = paper_data["title"] if paper_data["title"] else ""
    if type(abstract) == list:
        abstract = " ".join([a for a in abstract if a != None])
    text = abstract + title
    return text


def main(args):
    df = pd.read_csv(args.input).dropna(subset=["paper_data"])
    paper_titles = []
    paper_ids = []
    examples = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        paper_data = json.loads(row["paper_data"])
        paper_id = paper_data["paper_id"]
        acronym = row["acronym"]

        sents = get_examples(row["text"], acronym, paper_data, args.max_examples)
        examples.append([row["text"]] + sents)

        seen_papers = set()
        process_paper(row["text"], acronym, paper_data, paper_id, seen_papers, args.levels)
        paper_texts = [get_paper_text(es.get(index="s2orc", id=id)["_source"]) for id in seen_papers]
        paper_titles.append(paper_texts)
        paper_ids.append(paper_id)
    
    df["examples"] = examples
    df["paper_titles"] = paper_titles
    df["paper_id"] = paper_ids
    df.drop(columns=["paper_data"], inplace=True)
    filename = args.input.split(".")[0] + f"_{args.levels}_{args.max_examples}.csv"
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--levels", type=int, required=True)
    parser.add_argument("--max_examples", type=int, required=True)
    args = parser.parse_args()
    main(args)
