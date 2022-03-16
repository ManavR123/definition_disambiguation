import numpy as np
import torch
from elasticsearch import Elasticsearch, NotFoundError

from modeling.sgc import get_examples
from modeling.utils_modeling import get_embeddings


def get_baseline_embedding(model, tokenizer, acronym, device, text, embedding_mode):
    return get_embeddings(model, tokenizer, acronym, [text], device, embedding_mode)[0].cpu().numpy()


def process_paper(text, acronym, paper_data, paper_id, sents, seen_papers, levels, MAX_EXAMPLES):
    es = Elasticsearch(hosts=["http://localhost:9200"], timeout=60, retry_on_timeout=True)
    sents.extend(get_examples(text, acronym, paper_data, MAX_EXAMPLES))
    seen_papers.add(paper_id)

    if levels <= 0:
        return

    for cite in paper_data["inbound_citations"] + paper_data["outbound_citations"]:
        if cite in seen_papers:
            continue
        try:
            cite_data = es.get(index="s2orc", id=cite)["_source"]
            process_paper(text, acronym, cite_data, cite, sents, seen_papers, levels - 1, MAX_EXAMPLES)
        except NotFoundError:
            print(f"Could not find {cite}")


def get_average_embedding(model, tokenizer, device, acronym, paper_data, text, levels, MAX_EXAMPLES, embedding_mode):
    sents = [text]
    seen_papers = set()

    paper_id = paper_data["paper_id"]
    process_paper(text, acronym, paper_data, paper_id, sents, seen_papers, levels, MAX_EXAMPLES)

    X = []
    BATCH_SIZE = 32
    for i in range(0, len(sents), BATCH_SIZE):
        batch = sents[i : i + BATCH_SIZE]
        embeddings = get_embeddings(model, tokenizer, acronym, batch, device, embedding_mode).cpu().numpy()
        X.extend(embeddings)

    X = np.array(X)
    target = torch.tensor(X.mean(0))
    return target, sents
