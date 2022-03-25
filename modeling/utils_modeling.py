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
        tokens = word_tokenize(sent.replace("-", " "))
        idx = None
        for i, token in enumerate(tokens):
            if token.lower() == acronym.lower():
                idx = i
                break
        word_idx.append(idx)
    return word_idx


def get_word_mask(inputs, device, word_idx):
    mask = np.zeros((inputs["input_ids"].size(0), inputs["input_ids"].size(1)))
    for i, idx in enumerate(word_idx):
        if idx is not None:
            mask[i] = np.array(inputs.word_ids(i)) == idx
        else:
            mask[i] = inputs["attention_mask"][i].cpu().numpy()
    return torch.Tensor(mask).to(device)


def pool_embeddings(mode, acronym, sents, inputs, result, device):
    if mode == "CLS":
        embeds = result.last_hidden_state[:, 0, :]
    elif mode == "mean":
        mask = inputs["attention_mask"]
        embeds = mask_embeds(result.last_hidden_state, mask)
    elif mode == "acronym":
        word_idx = get_word_idx(acronym, sents)
        mask = get_word_mask(inputs, device, word_idx)
        embeds = mask_embeds(result.last_hidden_state, mask)
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


def embed_sents(model, tokenizer, device, acronym, embedding_mode, sents):
    X = []
    BATCH_SIZE = 32
    for i in range(0, len(sents), BATCH_SIZE):
        batch = sents[i : i + BATCH_SIZE]
        embeddings = get_embeddings(model, tokenizer, acronym, batch, device, embedding_mode).cpu().numpy()
        X.extend(embeddings)
    X = np.array(X)
    return X
