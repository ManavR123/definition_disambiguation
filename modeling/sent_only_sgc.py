from collections import deque
import itertools
from elasticsearch import Elasticsearch
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from modeling.sgc import aug_normalized_adjacency, get_examples, pool_features, sparse_mx_to_torch_sparse_tensor
from modeling.utils_modeling import embed_sents, get_embeddings

class PaperNode:
    def __init__(self, paper_id, sents, G):
        self.paper_id = paper_id
        self.sents = sents
        self.G = G

        self.node_ids = [f"{paper_id}-{i}" for i in range(len(sents))]
        for node_id, sent in zip(self.node_ids, sents):
            self.G.add_node(node_id, text=sent)
        self.G.add_edges_from(itertools.combinations(self.node_ids, 2))
    
    def add_parent(self, parent):
        assert self.G is parent.G
        self.G.add_edges_from(itertools.product(parent.node_ids, self.node_ids))

def create_graph(acronym, paper_data, text, levels, MAX_EXAMPLES):
    es = Elasticsearch(hosts=["http://localhost:9200"], timeout=60, retry_on_timeout=True)
    G = nx.Graph()
    seen_papers = set()
    
    q = deque([(paper_data, None, levels)])
    prev = [text]
    while len(q) > 0:
        paper_data, parent, levels = q.popleft()
        paper_id = paper_data["paper_id"]
        seen_papers.add(paper_id)
        pn = PaperNode(paper_id, prev + get_examples(text, acronym, paper_data, MAX_EXAMPLES), G)
        prev = []

        if parent:
            pn.add_parent(parent)
        if levels <= 0:
            continue
        
        for cite in paper_data["inbound_citations"] + paper_data["outbound_citations"]:
            if cite in seen_papers:
                continue
            cite_data = es.get(index="s2orc", id=cite)["_source"]
            q.append((cite_data, pn, levels - 1))
    return G

def get_sent_sgc_embedding(model, tokenizer, device, acronym, paper_data, text, k, levels, MAX_EXAMPLES, embedding_mode):
    G = create_graph(acronym, paper_data, text, levels, MAX_EXAMPLES)
    ids, sents = zip(*nx.get_node_attributes(G, "text").items())
    X = embed_sents(model, tokenizer, device, acronym, embedding_mode, list(sents))

    adj = nx.adjacency_matrix(G)
    S = aug_normalized_adjacency(adj)
    S = sparse_mx_to_torch_sparse_tensor(S)

    X = torch.FloatTensor(X).float()
    X = F.normalize(X, p=2, dim=1)
    Y = pool_features(X, S, k)
    target = Y[0].detach()
    return target, G
