from torch import nn
import torch
from transformers import AutoModel, AutoTokenizer

from modeling.utils_modeling import pool_embeddings


def create_input(tokenizer, contexts, glosses, device):
    encoded_contexts = tokenizer(contexts, return_tensors="pt", padding=True, truncation=True).to(device)
    encoded_glosses = tokenizer(glosses, return_tensors="pt", padding=True, truncation=True).to(device)
    return encoded_contexts, encoded_glosses


def get_scores(encoded_contexts, context_embeddings, gloss_embeddings, sents, words, glosses, device):
    gloss_embeddings = pool_embeddings(gloss_embeddings, "CLS", device)

    context_embeddings = pool_embeddings(context_embeddings, "acronym", device, words, sents, encoded_contexts)
    repeats = torch.Tensor([len(gs) for gs in glosses]).int().to(device)
    context_embeddings = torch.repeat_interleave(context_embeddings, repeats, dim=0)

    scores = torch.sum(context_embeddings * gloss_embeddings, dim=-1)
    return scores


class BEM(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(BEM, self).__init__()
        self.context_encoder = AutoModel.from_pretrained(model_name)
        self.gloss_encoder = self.context_encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, encoded_contexts, encoded_glosses):
        context_embeddings = self.context_encoder(**encoded_contexts)
        gloss_embeddings = self.gloss_encoder(**encoded_glosses)
        return context_embeddings, gloss_embeddings
