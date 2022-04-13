import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel

def main():
    df = pd.read_csv("pseudowords/terms.tsv", sep="\t")
    model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").eval().to("cuda:1")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

    summaries = df["summary"].tolist()

    embeds = []
    for i in tqdm(range(0, len(summaries), 32)):
        batch = summaries[i:i + 32]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to("cuda:1")
        with torch.no_grad():
            result = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"]
        mask = mask.unsqueeze(-1).expand(result.size()).float()
        embed = torch.sum(result * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        embeds.extend(embed.cpu().numpy())

    term_embeddings = {}
    term_to_summary = {}
    for i, term in enumerate(df["term"].tolist()):
        term_embeddings[term] = embeds[i]
        term_to_summary[term] = summaries[i]
    
    output = {
        "expansion_embeddings": term_embeddings,
        "expansion_to_sents": term_to_summary
    }
    np.save("pseudowords/terms_embed.npy", output)

if __name__ == "__main__":
    main()