import json
import pickle

import pandas as pd

paper_ids = set()
for file in ["train.csv", "dev.csv", "test.csv"]:
    df = pd.read_csv(file).dropna(subset=["paper_data"])
    for paper_data in df["paper_data"]:
        paper_data = json.loads(paper_data)
        paper_ids.update([paper_data["paper_id"]])
        paper_ids.update(paper_data["inbound_citations"])
        paper_ids.update(paper_data["outbound_citations"])

print(len(paper_ids))
with open("paper_ids.pickle", "wb") as f:
    pickle.dump(paper_ids, f)
