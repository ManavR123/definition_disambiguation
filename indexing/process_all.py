"""Script to find all of the paper ids that each sample in the SciAD dataset belongs to.

Downloads each metadata and pdf_parse file one at a time, creates an index over them, then searches the index for each
sample. The found results are written to the original dataset file.

The index is then deleted and the downloaded files are removed.
"""
import json
import os
import subprocess
import urllib.request

import pandas as pd
from tqdm import tqdm

from indexing.utils_index import add_shard_to_index, create_index, delete_index, search

metadata_urls = open("metadata_urls.txt").read().split("\n")
pdf_parse_urls = open("pdf_parse_urls.txt").read().split("\n")


def search_file(file):
    """Search the index for each sample in the given file. Write the results to the original dataset file."""
    print(f"Searching {file}")
    df = pd.read_csv(file)
    paper_data = df["paper_data"].tolist()
    count = 0
    for i in tqdm(range(df.shape[0])):
        results = search(df.iloc[i]["text"])
        if len(results) == 0:
            continue

        paper_data[i] = json.dumps(results[0])  # save top result
        count += 1

    print(f"Found {count} results\n")
    df["paper_data"] = paper_data
    df.to_csv(file, index=False)


def process_shard(id):
    """Processes a single shard of the s2orc dataset."""
    create_index()
    add_shard_to_index(id)

    for file in ["train.csv", "dev.csv"]:
        search_file(file)

    delete_index()


def unzip_data(path):
    cmd = ["gzip", "-d", path]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def process_all():
    for i in tqdm(range(0, 100)):
        metadata_path = f"20200705v1/full/metadata/metadata_{i}.jsonl.gz"
        parse_path = f"20200705v1/full/pdf_parses/pdf_parses_{i}.jsonl.gz"

        # download files
        urllib.request.urlretrieve(metadata_urls[i], metadata_path)
        urllib.request.urlretrieve(pdf_parse_urls[i], parse_path)

        # unzip files
        unzip_data(metadata_path)
        unzip_data(parse_path)

        process_shard(i)

        # delete files
        os.remove(metadata_path.replace(".gz", ""))
        os.remove(parse_path.replace(".gz", ""))


if __name__ == "__main__":
    process_all()
