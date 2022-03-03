"""Script to add paper metadata and pdf_parses to an elastic search index"""
import json
import time

import tqdm
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

es = Elasticsearch(hosts=["http://localhost:9200"], timeout=60, retry_on_timeout=True)

# Add the paper metadata to the index first then update each entry with the pdf_parses
# not all papers have pdf_parses so it makes sense to add the metadata first
print("Loading metadata")
with open(f"indexing/metadata.jsonl") as f_meta:
    rows = []
    for i, line in tqdm(enumerate(f_meta)):
        metadata_dict = json.loads(line)
        data = {"_id": metadata_dict["paper_id"], "_index": "s2orc"}
        data.update(metadata_dict)

        rows.append(data)

        if i % 100000 == 0:
            print(f"\nAdding batch")
            start = time.time()
            helpers.bulk(es, rows)
            print(f"Added batch in {(time.time() - start) / 60} minutes")
            rows = []

print(f"\nAdding batch")
start = time.time()
helpers.bulk(es, rows)
print(f"Added batch in {(time.time() - start) / 60} minutes")

print("Loading pdf_parses")
with open(f"indexing/pdf_parses.jsonl") as f_pdf:
    for i, line in tqdm(enumerate(f_pdf)):
        pdf_parse_dict = json.loads(line)
        data = {}
        data["pdf_parse"] = [p["text"] for p in pdf_parse_dict["body_text"]]
        es.update(index="s2orc", doc_type="_doc", id=pdf_parse_dict["paper_id"], body={"doc": data})

# flush index
es.indices.refresh(index="s2orc")
es.indices.flush(index="s2orc")
