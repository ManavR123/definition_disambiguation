import torch
from elasticsearch import Elasticsearch, NotFoundError
from transformers import AutoModel, AutoTokenizer

from modeling.sgc import get_examples, get_paper_text
from modeling.utils_modeling import embed_sents, get_embeddings

es = Elasticsearch(hosts=["http://localhost:9200"], timeout=60, retry_on_timeout=True)

paper_model = AutoModel.from_pretrained("allenai/specter").eval()
paper_tokenizer = AutoTokenizer.from_pretrained("allenai/specter")


def get_baseline_embedding(model, tokenizer, acronym, device, text, embedding_mode):
    return get_embeddings(model, tokenizer, acronym, [text], device, embedding_mode).squeeze()


def process_paper(text, acronym, paper_data, paper_id, sents, seen_papers, levels, MAX_EXAMPLES):
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
    sents, x, _ = average_embedding_helper(
        model, tokenizer, device, acronym, paper_data, text, levels, MAX_EXAMPLES, embedding_mode
    )
    return x, sents


def get_paper_average_embedding(
    model, tokenizer, device, acronym, paper_data, text, levels, MAX_EXAMPLES, embedding_mode
):
    sents, x, z = average_embedding_helper(
        model, tokenizer, device, acronym, paper_data, text, levels, MAX_EXAMPLES, embedding_mode
    )
    target = torch.cat((x, z))
    return target, sents


def average_embedding_helper(model, tokenizer, device, acronym, paper_data, text, levels, MAX_EXAMPLES, embedding_mode):
    sents = [text]
    seen_papers = set()
    paper_model.to(device)

    paper_id = paper_data["paper_id"]
    process_paper(text, acronym, paper_data, paper_id, sents, seen_papers, levels, MAX_EXAMPLES)

    X = embed_sents(model, tokenizer, device, acronym, embedding_mode, sents)

    paper_texts = [get_paper_text(es.get(index="s2orc", id=id)["_source"]) for id in seen_papers]
    sents += paper_texts

    Z = embed_sents(paper_model, paper_tokenizer, device, acronym, "CLS", paper_texts)

    x, z = torch.tensor(X.mean(0)), torch.tensor(Z.mean(0))
    x, z = x / torch.norm(x), z / torch.norm(z)
    return sents, x, z
