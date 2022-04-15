import json
import time

from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

es = Elasticsearch(hosts=["http://localhost:9200"], timeout=60, retry_on_timeout=True)


def create_index():
    """Creates the s2orc index"""
    es.indices.create(
        index="s2orc",
        settings={"similarity": {"default": {"type": "BM25"}}},
        mappings={
            "properties": {
                "title": {"type": "text", "index": True},
                "abstract": {"type": "text", "index": True},
                "pdf_parse": {"type": "text", "index": True},
                "paper_id": {"type": "text", "index": True},
                "arxiv_id": {"type": "text", "index": True},
                "acl_id": {"type": "text", "index": False},
                "pmc_id": {"type": "text", "index": False},
                "pubmed_id": {"type": "text", "index": False},
                "mag_id": {"type": "text", "index": False},
                "mag_field_of_study": {"type": "text", "index": False},
                "outbound_citations": {"type": "text", "index": False},
                "inbound_citations": {"type": "text", "index": False},
                "year": {"type": "text", "index": False},
                "doi": {"type": "text", "index": False},
                "venue": {"type": "text", "index": False},
                "journal": {"type": "text", "index": False},
                "has_outbound_citations": {"type": "boolean", "index": False},
                "has_inbound_citations": {"type": "boolean", "index": False},
                "has_pdf_parse": {"type": "boolean", "index": False},
                "s2_url": {"type": "text", "index": False},
            }
        },
    )
    print("Index created")


def add_shard_to_index(id):
    """Adds a shard of the s2orc dataset to the index"""

    # load the paper_parses, map paper_id to paper_parse
    print("Loading pdf_parses")
    paper_id_to_pdf_parse = {}
    with open(f"20200705v1/full/pdf_parses/pdf_parses_{id}.jsonl") as f_pdf:
        for line in tqdm(f_pdf):
            pdf_parse_dict = json.loads(line)
            paper_id_to_pdf_parse[pdf_parse_dict["paper_id"]] = pdf_parse_dict

    print("Loading metadata")
    with open(f"20200705v1/full/metadata/metadata_{id}.jsonl") as f_meta:
        # for each paper in the metadata file, create a dict of the paper's metadata and add its paper_parse
        rows = []
        for line in tqdm(f_meta):
            metadata_dict = json.loads(line)
            data = {"_id": metadata_dict["paper_id"], "_index": "s2orc"}
            data.update(metadata_dict)

            paper_id = metadata_dict["paper_id"]
            if paper_id in paper_id_to_pdf_parse:
                if len(pdf_parse_dict["abstract"]) > 0:
                    data["abstract"] = [data["abstract"], pdf_parse_dict["abstract"][0]["text"]]
                data["pdf_parse"] = [
                    p["text"] for p in paper_id_to_pdf_parse.get(paper_id, {"body_text": []})["body_text"]
                ]
            rows.append(data)

        # add the papers to the index
        print(f"\nAdding {id}")
        start = time.time()
        helpers.bulk(es, rows)
        print(f"Added {id} in {(time.time() - start) / 60} minutes")

    # flush index
    es.indices.refresh(index="s2orc")
    es.indices.flush(index="s2orc")


def delete_index():
    """Deletes the s2orc index"""
    es.indices.delete(index="s2orc")
    print("Index deleted")


def search(text_query, fields=None, limit=1):
    """Performs a phrase query across the title, abstract, and pdf_parse fields

    Args:
        text_query (str): The query to search for
        limit (int): The number of results to return

    Returns:
        list: A list of dicts containing the found papers
    """
    if fields is None:
        fields = ["title", "abstract", "pdf_parse"]
    query = {
        "multi_match": {
            "query": f'"{text_query}"',
            "fields": fields,
            "type": "phrase",
            "operator": "or",
        }
    }

    return [result["_source"] for result in es.search(index="s2orc", size=limit, query=query)["hits"]["hits"]]
