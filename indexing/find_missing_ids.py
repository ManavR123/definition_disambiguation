"""If somehow some papers weren't added to the index, this script will which ones weren't added"""
import json
import pickle

import pandas as pd
from elasticsearch import Elasticsearch, NotFoundError
from tqdm import tqdm

es = Elasticsearch(hosts=["http://localhost:9200"], timeout=60, retry_on_timeout=True)

missed_paper_ids = set()  # set of paper ids that were not found in the index
cites = set()  # cites of papers that were found in the index, need to make sure these are in the index too
searched = set()  # papers that were searched for

# first check papers and their cites
for file in ["train.csv", "dev.csv", "test.csv"]:
    df = pd.read_csv(f"sciad_data/{file}").dropna(subset=["paper_data"])
    for i in tqdm(range(len(df))):
        paper_data = json.loads(df.iloc[i]["paper_data"])
        paper_id = paper_data["paper_id"]

        # check if paper and its cites are in the index
        new_cites = paper_data["inbound_citations"] + paper_data["outbound_citations"]
        for cite in [paper_id] + new_cites:
            if cite in searched:
                continue
            searched.add(cite)
            try:  # check if paper exists in index
                cite_data = es.get(index="s2orc", id=cite)["_source"]
                # store cites of cites
                cites.update(cite_data["inbound_citations"] + cite_data["outbound_citations"])
            except NotFoundError:
                missed_paper_ids.add(cite)

# now check if cites of cites are in the index
for cite in tqdm(cites):
    if cite in searched:
        continue
    try:
        es.get(index="s2orc", id=cite)
    except NotFoundError:
        missed_paper_ids.add(cite)

print(f"Found {len(missed_paper_ids)} papers that were not found in the index")
with open("missed_paper_ids.pickle", "wb") as f:
    pickle.dump(missed_paper_ids, f)
