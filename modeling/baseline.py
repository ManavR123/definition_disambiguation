import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from modeling.sgc import add_embeddings_to_graph, process_paper

from modeling.utils_modeling import get_embeddings


def get_baseline_embedding(model, tokenizer, acronym, device, text, embedding_mode):
    return get_embeddings(model, tokenizer, acronym, [text], device, embedding_mode)[0].cpu().numpy()

def get_average_embedding(model, tokenizer, device, acronym, paper_data, text, levels, MAX_EXAMPLES, embedding_mode):
    G = nx.Graph()
    data = {}

    paper_id = paper_data["paper_id"]
    G.add_node(f"{paper_id}-text")
    data[f"{paper_id}-text"] = text

    process_paper(G, text, acronym, paper_data, paper_id, f"{paper_id}-text", data, levels, MAX_EXAMPLES)
    add_embeddings_to_graph(G, data, acronym, model, tokenizer, device, embedding_mode)

    X = np.array(list(nx.get_node_attributes(G, "embedding").values()))
    X = torch.FloatTensor(X).float()
    target = X.mean(0).detach().numpy()
    return target, G