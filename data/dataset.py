from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset


class WSDDataset(Dataset):
    def __init__(self, train_data: pd.DataFrame, word_to_senses: Dict[str, str], sense_to_gloss: pd.DataFrame):
        data = train_data
        word_to_senses = word_to_senses
        sense_to_gloss = {row["term"]: row["summary"] for _, row in sense_to_gloss.iterrows()}

        self.batches = []
        for _, row in tqdm(data.iterrows(), total=len(data)):
            text = row["text"]
            acronym = row["acronym"]
            labels = torch.Tensor([sense == row["expansion"] for sense in word_to_senses[acronym]]).float()
            glosses = [sense_to_gloss[sense] for sense in word_to_senses[acronym]]
            self.batches.append(
                {
                    "text": text,
                    "acronym": acronym,
                    "labels": labels,
                    "glosses": glosses,
                    "expansion": row["expansion"],
                    "paper_titles": eval(row["paper_titles"])[0],
                    "paper_id": row["paper_id"],
                }
            )

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx: int):
        return self.batches[idx]

    @staticmethod
    def collate_fn(batch: Dict[str, List[any]]):
        return {
            "text": [item["text"] for item in batch],
            "acronym": [item["acronym"] for item in batch],
            "labels": torch.cat([item["labels"] for item in batch], dim=0),
            "glosses": [item["glosses"] for item in batch],
            "paper_titles": [item["paper_titles"] for item in batch],
        }
