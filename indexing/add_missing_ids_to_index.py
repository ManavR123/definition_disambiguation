"""Script to add the missing paper ids to the index."""
import gzip
import io
import json
import os
import pickle
import urllib.request
from collections import defaultdict
from time import time

from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

metadata_urls = open("metadata_urls.txt").read().split("\n")
pdf_parse_urls = open("pdf_parse_urls.txt").read().split("\n")
with open("missed_paper_ids.pickle", "rb") as f:
    paper_ids = pickle.load(f)

rows = defaultdict(dict)


def collect_papers(paper_ids, metadata_path, pdf_parse_path):
    collected_papers = set()
    with open("metadata.jsonl", "ab") as f_out:
        with gzip.open(metadata_path, "rb") as gz:
            f = io.BufferedReader(gz)
            for line in tqdm(f.readlines()):
                metadata_dict = json.loads(line)
                paper_id = metadata_dict["paper_id"]
                if paper_id in paper_ids:
                    rows[paper_id].update({"_id": metadata_dict["paper_id"], "_index": "s2orc"})
                    rows[paper_id].update(metadata_dict)
                    f_out.write(line)
                    collected_papers.add(paper_id)

    with open("pdf_parses.jsonl", "ab") as f_out:
        with gzip.open(pdf_parse_path, "rb") as gz:
            f = io.BufferedReader(gz)
            for line in tqdm(f.readlines()):
                pdf_parse_dict = json.loads(line)
                paper_id = pdf_parse_dict["paper_id"]
                if paper_id in paper_ids:
                    rows[paper_id]["pdf_parse"] = [p["text"] for p in pdf_parse_dict["body_text"]]
                    f_out.write(line)

    print(f"\n{len(collected_papers)} papers are collected")
    paper_ids.difference_update(collected_papers)


for i in tqdm(range(0, 100)):
    metadata_path = f"20200705v1/full/metadata/metadata_{i}.jsonl.gz"
    parse_path = f"20200705v1/full/pdf_parses/pdf_parses_{i}.jsonl.gz"

    print("\nDownloading {}".format(metadata_path))
    urllib.request.urlretrieve(metadata_urls[i], metadata_path)
    print("\nDownloading {}".format(parse_path))
    urllib.request.urlretrieve(pdf_parse_urls[i], parse_path)

    print("\Colecting Papers...")
    collect_papers(paper_ids, metadata_path, parse_path)

    print("Deleting files")
    os.remove(metadata_path)
    os.remove(parse_path)

    if len(paper_ids) == 0:
        print("\nNo new papers")
        break

print(f"\n{len(rows)}/{len(paper_ids)} new papers")

# add found papers to index
es = Elasticsearch(hosts=["http://localhost:9200"], timeout=60, retry_on_timeout=True)
print("\nUploading to Elasticsearch...")
start = time()
helpers.bulk(es, list(rows.values()))
print("\nTime taken: {} minutes".format((time() - start) / 60))
