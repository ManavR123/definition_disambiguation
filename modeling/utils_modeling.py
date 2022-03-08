from nltk.tokenize import word_tokenize
import numpy as np
import torch
import torch.nn.functional as F


def mask_embeds(token_embeddings, mask):
    mask = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    embeds = torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
    return embeds


def get_word_idx(acronym, sents):
    word_idx = []
    for sent in sents:
        tokens = word_tokenize(sent.lower().replace("-", " "))
        idx = None
        for i, token in enumerate(tokens):
            if token.lower() == acronym.lower():
                idx = i
                break
        word_idx.append(idx)
    return word_idx


def pool_embeddings(mode, acronym, sents, inputs, result, device):
    if mode == "CLS":
        embeds = result.last_hidden_state[:, 0, :]
    elif mode == "mean":
        mask = inputs["attention_mask"]
        embeds = mask_embeds(result.last_hidden_state, mask)
    elif mode == "acronym":
        word_idx = get_word_idx(acronym, sents)
        mask = torch.Tensor(np.array([np.array(inputs.word_ids(i)) == idx for i, idx in enumerate(word_idx)])).to(
            device
        )
        embeds = mask_embeds(result.last_hidden_state, mask)
    embeds = F.normalize(embeds, p=2, dim=1)
    return embeds


def get_embeddings(model, tokenizer, acronym, sents, device, mode, is_train=False):
    inputs = tokenizer(
        sents,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    ).to(device)

    if is_train:
        result = model(**inputs)
    else:
        with torch.no_grad():
            result = model(**inputs)

    return pool_embeddings(mode, acronym, sents, inputs, result, device)


def get_baseline_embedding(model, tokenizer, acronym, device, text):
    return get_embeddings(model, tokenizer, acronym, [text], device, "acronym")[0]
