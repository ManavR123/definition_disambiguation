import numpy as np
import torch


def mask_embeds(token_embeddings, mask):
    mask = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    embeds = torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
    return embeds


def get_word_idx(acronyms, inputs, sents):
    word_idx = []
    for i in range(len(sents)):
        max_id = max(inputs.word_ids(i), key=lambda x: x or -1)
        for id in range(max_id + 1):
            start, end = inputs.word_to_chars(i, id)
            if sents[i][start:end] == acronyms[i]:
                word_idx.append(id)
                break
        else:
            word_idx.append(None)
    return word_idx


def get_word_mask(inputs, device, word_idx):
    mask = np.zeros((inputs["input_ids"].size(0), inputs["input_ids"].size(1)))
    for i, idx in enumerate(word_idx):
        if idx is not None:
            mask[i] = np.array(inputs.word_ids(i)) == idx
        else:
            mask[i] = inputs["attention_mask"][i].cpu().numpy()
    return torch.Tensor(mask).to(device)


def pool_embeddings(result, mode, device=None, acronyms=None, sents=None, inputs=None):
    if mode == "CLS":
        embeds = result.last_hidden_state[:, 0, :]
    elif mode == "mean":
        mask = inputs["attention_mask"]
        embeds = mask_embeds(result.last_hidden_state, mask)
    elif mode == "acronym":
        word_idx = get_word_idx(acronyms, inputs, sents)
        mask = get_word_mask(inputs, device, word_idx)
        embeds = mask_embeds(result.last_hidden_state, mask)
    return embeds


def get_embeddings(model, tokenizer, acronyms, sents, device, mode, is_train=False):
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

    return pool_embeddings(result, mode, device, acronyms, sents, inputs)


def embed_sents(model, tokenizer, device, acronyms, embedding_mode, sents):
    X = []
    BATCH_SIZE = 32
    for i in range(0, len(sents), BATCH_SIZE):
        batch = sents[i : i + BATCH_SIZE]
        embeddings = get_embeddings(model, tokenizer, acronyms[i : i + BATCH_SIZE], batch, device, embedding_mode).cpu().numpy()
        X.extend(embeddings)
    X = np.array(X)
    return X
