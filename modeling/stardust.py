import itertools
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from modeling.utils_modeling import pool_embeddings
from modeling.wsd_model import WSDModel


def flatten(l):
    return list(itertools.chain(*l))


def average_embeddings(all_embeddings, batch_texts):
    batch_embeddings = []
    for texts in batch_texts:
        batch_embeddings.append(all_embeddings[len(batch_embeddings) : len(batch_embeddings) + len(texts)].mean(dim=0))
    return torch.stack(batch_embeddings)


class Stardust(WSDModel):
    def __init__(
        self,
        model_name="bert-base-uncased",
        hidden_size=128,
        tie_context_gloss_encoder=True,
        freeze_context_encoder=False,
        freeze_gloss_encoder=False,
        freeze_paper_encoder=False,
    ):
        super(Stardust, self).__init__()
        self.context_encoder = AutoModel.from_pretrained(model_name)
        if tie_context_gloss_encoder:
            self.gloss_encoder = self.context_encoder
        else:
            self.gloss_encoder = AutoModel.from_pretrained(model_name)
        self.paper_encoder = AutoModel.from_pretrained("allenai/specter")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.paper_tokenizer = AutoTokenizer.from_pretrained("allenai/specter")

        if freeze_context_encoder:
            for param in self.context_encoder.parameters():
                param.requires_grad = False
        if freeze_gloss_encoder:
            for param in self.gloss_encoder.parameters():
                param.requires_grad = False
        if freeze_paper_encoder:
            for param in self.paper_encoder.parameters():
                param.requires_grad = False

        input_size = (
            self.context_encoder.config.hidden_size
            + self.gloss_encoder.config.hidden_size
            + self.paper_encoder.config.hidden_size
        )
        self.model = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, encoded_contexts, encoded_glosses, encoded_papers):
        context_embeddings = self.context_encoder(**encoded_contexts)
        gloss_embeddings = self.gloss_encoder(**encoded_glosses)
        paper_embeddings = self.paper_encoder(**encoded_papers)
        return context_embeddings, gloss_embeddings, paper_embeddings

    def create_input(self, contexts, glosses, papers, device):
        encoded_contexts = self.tokenizer(contexts, return_tensors="pt", padding=True, truncation=True).to(device)
        encoded_glosses = self.tokenizer(glosses, return_tensors="pt", padding=True, truncation=True).to(device)
        encoded_papers = self.paper_tokenizer(papers, return_tensors="pt", padding=True, truncation=True).to(device)
        return encoded_contexts, encoded_glosses, encoded_papers

    def get_scores(
        self,
        context_embeddings,
        gloss_embeddings,
        paper_embeddings,
        encoded_contexts,
        encoded_glosses,
        sents,
        words,
        glosses,
        paper_titles,
        device,
    ):
        gloss_embeddings = pool_embeddings(gloss_embeddings, "mean", device, inputs=encoded_glosses)
        repeats = torch.Tensor([len(gs) for gs in glosses]).int().to(device)

        context_embeddings = pool_embeddings(
            context_embeddings,
            "acronym",
            device,
            flatten([[word] * len(sents[i]) for i, word in enumerate(words)]),
            flatten(sents),
            encoded_contexts,
        )
        context_embeddings = average_embeddings(context_embeddings, sents)
        context_embeddings = torch.repeat_interleave(context_embeddings, repeats, dim=0)

        paper_embeddings = pool_embeddings(paper_embeddings, "CLS", device)
        paper_embeddings = average_embeddings(paper_embeddings, paper_titles)
        paper_embeddings = torch.repeat_interleave(paper_embeddings, repeats, dim=0)

        x = torch.cat([context_embeddings, gloss_embeddings, paper_embeddings], dim=-1)
        return self.model(x).squeeze(-1)

    def step(self, batch, device):
        encoded_contexts, encoded_glosses, encoded_papers = self.create_input(
            flatten(batch["examples"]), flatten(batch["glosses"]), flatten(batch["paper_titles"]), device
        )
        context_embeddings, gloss_embeddings, paper_embeddings = self(encoded_contexts, encoded_glosses, encoded_papers)
        scores = self.get_scores(
            context_embeddings,
            gloss_embeddings,
            paper_embeddings,
            encoded_contexts,
            encoded_glosses,
            batch["examples"],
            batch["acronym"],
            batch["glosses"],
            batch["paper_titles"],
            device,
        )
        return scores
