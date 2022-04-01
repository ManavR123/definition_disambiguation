import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from elasticsearch import Elasticsearch, NotFoundError
from nltk.tokenize import sent_tokenize

from modeling.utils_modeling import get_embeddings


def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def add_embeddings_to_graph(G, data, acronym, model, tokenizer, device, embedding_mode, batch_size=256):
    nodes = list(data.keys())
    sents = [data[node] for node in nodes]

    for i in range(0, len(nodes), batch_size):
        batch = sents[i : i + batch_size]
        embeddings = get_embeddings(model, tokenizer, acronym, batch, device, embedding_mode).cpu().numpy()
        for j, node in enumerate(nodes[i : i + batch_size]):
            G.nodes[node]["sent"] = batch[j]
            G.nodes[node]["embedding"] = embeddings[j]


def pool_features(features, adj, degree):
    for _ in range(degree):
        features = torch.spmm(adj, features)
    return features


def get_paper_text(paper_data):
    abstract = paper_data["abstract"] if paper_data["abstract"] != None else ""
    title = paper_data["title"] if paper_data["title"] else ""
    if type(abstract) == list:
        abstract = " ".join([a for a in abstract if a != None])
    text = abstract + title
    return text


def get_examples(text, acronym, paper_data, MAX_EXAMPLES):
    examples = []
    for para in paper_data.get("pdf_parse", []):
        sents = sent_tokenize(para)
        for sent in sents:
            if len(examples) >= MAX_EXAMPLES:
                return examples
            if acronym in sent and text not in sent:
                examples.append(sent)
    return examples


def add_examples(G, text, acronym, paper_id, paper_data, data, MAX_EXAMPLES):
    examples = get_examples(text, acronym, paper_data, MAX_EXAMPLES)
    for count, sent in enumerate(examples):
        G.add_node(f"{paper_id}-{count}")
        data[f"{paper_id}-{count}"] = sent
        G.add_edge(paper_id, f"{paper_id}-{count}")


def process_paper(G, text, acronym, paper_data, paper_id, parent_id, data, levels, MAX_EXAMPLES):
    es = Elasticsearch(hosts=["http://localhost:9200"], timeout=60, retry_on_timeout=True)
    G.add_node(paper_id)
    data[paper_id] = get_paper_text(paper_data)
    G.add_edge(paper_id, parent_id)
    add_examples(G, text, acronym, paper_id, paper_data, data, MAX_EXAMPLES)

    if levels <= 0:
        return

    for cite in paper_data["inbound_citations"] + paper_data["outbound_citations"]:
        if cite in G:
            continue
        try:
            cite_data = es.get(index="s2orc", id=cite)["_source"]
            process_paper(G, text, acronym, cite_data, cite, paper_id, data, levels - 1, MAX_EXAMPLES)
        except NotFoundError:
            print(f"Could not find {cite}")


def sgc(k, G, X):
    adj = nx.adjacency_matrix(G)
    S = aug_normalized_adjacency(adj)
    S = sparse_mx_to_torch_sparse_tensor(S)
    X = torch.FloatTensor(X).float()
    X = F.normalize(X, p=2, dim=1)
    Y = pool_features(X, S, k)
    target = Y[0].detach()
    target = target / torch.norm(target)
    return target


def get_sgc_embedding(model, tokenizer, device, acronym, paper_data, text, k, levels, MAX_EXAMPLES, embedding_mode):
    G = nx.Graph()
    data = {}

    paper_id = paper_data["paper_id"]
    G.add_node(f"{paper_id}-text")
    data[f"{paper_id}-text"] = text

    process_paper(G, text, acronym, paper_data, paper_id, f"{paper_id}-text", data, levels, MAX_EXAMPLES)
    add_embeddings_to_graph(G, data, acronym, model, tokenizer, device, embedding_mode)
    X = np.array(list(nx.get_node_attributes(G, "embedding").values()))

    target = sgc(k, G, X)
    return target, G
