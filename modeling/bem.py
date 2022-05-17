import itertools

import torch
from transformers import AutoModel, AutoTokenizer

from modeling.utils_modeling import pool_embeddings
from modeling.wsd_model import WSDModel


class BEM(WSDModel):
    def __init__(self, model_name="bert-base-uncased"):
        super(BEM, self).__init__()
        self.context_encoder = AutoModel.from_pretrained(model_name)
        self.gloss_encoder = self.context_encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, encoded_contexts, encoded_glosses):
        context_embeddings = self.context_encoder(**encoded_contexts)
        gloss_embeddings = self.gloss_encoder(**encoded_glosses)
        return context_embeddings, gloss_embeddings

    def create_input(self, contexts, glosses, device):
        encoded_contexts = self.tokenizer(contexts, return_tensors="pt", padding=True, truncation=True).to(device)
        encoded_glosses = self.tokenizer(glosses, return_tensors="pt", padding=True, truncation=True).to(device)
        return encoded_contexts, encoded_glosses

    def get_scores(self, encoded_contexts, context_embeddings, gloss_embeddings, sents, words, glosses, device):
        gloss_embeddings = pool_embeddings(gloss_embeddings, "CLS", device)
        context_embeddings = pool_embeddings(context_embeddings, "acronym", device, words, sents, encoded_contexts)
        repeats = torch.Tensor([len(gs) for gs in glosses]).int().to(device)
        context_embeddings = torch.repeat_interleave(context_embeddings, repeats, dim=0)

        scores = torch.sum(context_embeddings * gloss_embeddings, dim=-1)
        return scores

    def step(self, batch, device):
        encoded_contexts, encoded_glosses = self.create_input(
            batch["text"], list(itertools.chain(*batch["glosses"])), device
        )
        context_embeddings, gloss_embeddings = self(encoded_contexts, encoded_glosses)
        scores = self.get_scores(
            encoded_contexts,
            context_embeddings,
            gloss_embeddings,
            batch["text"],
            batch["acronym"],
            batch["glosses"],
            device,
        )
        return scores
