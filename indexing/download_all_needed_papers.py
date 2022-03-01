"""Processes each shard of the s2orc dataset. Stores all papers needed for the SciAD dataset in a single file."""
import gzip
import io
import json
import os
import pickle
import urllib.request

from tqdm import tqdm

metadata_urls = open("metadata_urls.txt").read().split("\n")
pdf_parse_urls = open("pdf_parse_urls.txt").read().split("\n")

# createw new jsonl files to store the relevant papers
with open("metadata.jsonl", "w") as f:
    pass

with open("pdf_parses.jsonl", "w") as f:
    pass


def collect_papers(paper_ids, metadata_path, pdf_parse_path):
    """Processes a shard of the s2orc dataset. Finds all papers in paper_ids and
    adds them to the metadata and pdf_parse files

    Args:
        paper_ids (set): set of paper_ids to store
        metadata_path (str): path to shard of s2orc to process
        pdf_parse_path (str): path to shard of s2orc to process

    Returns:
        new_paper_ids (set): set of paper_ids that are connected to papers in paper_ids, but not in paper_ids

    Side Effects:
        writes found papers to metadata and pdf_parse files
        prints the number of found papers
    """
    new_papers_ids = set()
    count = 0
    with open("metadata.jsonl", "ab") as f_out:
        with gzip.open(metadata_path, "rb") as gz:
            f = io.BufferedReader(gz)
            for line in tqdm(f.readlines()):
                metadata_dict = json.loads(line)
                paper_id = metadata_dict["paper_id"]
                if paper_id in paper_ids:
                    f_out.write(line)
                    count += 1
                    for p_id in metadata_dict["outbound_citations"] + metadata_dict["inbound_citations"]:
                        if p_id not in paper_ids:
                            new_papers_ids.add(p_id)

    with open("pdf_parses.jsonl", "ab") as f_out:
        with gzip.open(pdf_parse_path, "rb") as gz:
            f = io.BufferedReader(gz)
            for line in tqdm(f.readlines()):
                metadata_dict = json.loads(line)
                paper_id = metadata_dict["paper_id"]
                if paper_id in paper_ids:
                    f_out.write(line)

    print("\n{} papers are collected".format(count))
    return new_papers_ids


def process_s2orc(paper_ids):
    """Processes the s2orc dataset to find papers that are in paper_ids"""
    new_paper_ids = set()
    for i in tqdm(range(0, 100)):
        metadata_path = f"20200705v1/full/metadata/metadata_{i}.jsonl.gz"
        parse_path = f"20200705v1/full/pdf_parses/pdf_parses_{i}.jsonl.gz"

        print("\nDownloading {}".format(metadata_path))
        urllib.request.urlretrieve(metadata_urls[i], metadata_path)
        print("\nDownloading {}".format(parse_path))
        urllib.request.urlretrieve(pdf_parse_urls[i], parse_path)

        print("\Colecting Papers...")
        temp = collect_papers(paper_ids, metadata_path, parse_path)
        new_paper_ids.update(temp)

        print("Deleting files")
        os.remove(metadata_path)
        os.remove(parse_path)
    return new_paper_ids


if __name__ == "__main__":
    # read in paper_ids.pickle - from collect_needed_paper_ids.py
    with open("paper_ids.pickle", "rb") as f:
        paper_ids = pickle.load(f)

    # first process all of the papers and their citations
    new_paper_ids = process_s2orc(paper_ids)
    # new_paper_ids has the citations of the citations, we need to store these too
    process_s2orc(new_paper_ids)
