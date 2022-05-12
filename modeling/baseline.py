import torch
from transformers import AutoModel, AutoTokenizer

from modeling.utils_modeling import embed_sents

paper_model = AutoModel.from_pretrained("allenai/specter").eval()
paper_tokenizer = AutoTokenizer.from_pretrained("allenai/specter")

def get_average_embedding(model, tokenizer, device, acronym, examples, embedding_mode):
    X = embed_sents(model, tokenizer, device, [acronym] * len(examples), embedding_mode, examples)
    x = torch.tensor(X.mean(0))
    x = x / torch.norm(x)
    return x, examples


def get_paper_average_embedding(model, tokenizer, device, acronym, examples, paper_titles, embedding_mode):
    paper_model.to(device)
    X = embed_sents(model, tokenizer, device, [acronym] * len(examples), embedding_mode, examples)
    Z = embed_sents(paper_model, paper_tokenizer, device, [acronym] * len(examples), "CLS", paper_titles)
    x, z = torch.tensor(X.mean(0)), torch.tensor(Z.mean(0))
    x, z = x / torch.norm(x), z / torch.norm(z)

    target = torch.cat((x, z))
    return target, examples + paper_titles
