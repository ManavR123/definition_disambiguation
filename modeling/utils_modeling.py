import torch
import torch.nn.functional as F


def get_baseline_embedding(model, tokenizer, device, text):
    return get_embeddings(model, tokenizer, [text], device)[0]


def get_embeddings(model, tokenizer, batch, device):
    inputs = tokenizer(
        batch,
        truncation=True,
        return_tensors="pt",
        max_length=512,
        padding="max_length",
    ).to(device)
    with torch.no_grad():
        result = model(**inputs)
    token_embeddings = result[0]
    input_mask_expanded = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
    embeds = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embeds = F.normalize(embeds, p=2, dim=1).detach().cpu().numpy()
    return embeds
