import json

import pandas as pd
import tqdm
from nltk.tokenize.treebank import TreebankWordDetokenizer


def convert_format(file):
    with open(file, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    new_data = []
    for d in tqdm.tqdm(data):
        acronym_idx = d["acronym"]
        tokens = (
            d["tokens"][:acronym_idx]
            + ["</s>"]
            + [d["tokens"][acronym_idx]]
            + ["</s>"]
            + d["tokens"][acronym_idx + 1 :]
        )
        sent = TreebankWordDetokenizer().detokenize(d["tokens"]).replace(" - ", "-")
        annotated_sent = TreebankWordDetokenizer().detokenize(tokens).replace(" - ", "-")
        new_data.append([sent, d["tokens"][acronym_idx], annotated_sent, d.get("expansion", ""), d.get("id", ""), ""])

    df = pd.DataFrame(new_data)
    df.to_csv(
        file.replace(".json", ".csv"),
        index=False,
        header=["text", "acronym", "annotated_text", "expansion", "sample_id", "paper_data"],
    )


for file in ["train.json", "dev.json"]:
    convert_format(file)
