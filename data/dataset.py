from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset


class WSDDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        word_to_senses: Dict[str, List[str]],
        sense_to_gloss: pd.DataFrame,
        reduce_dict: bool = False,
        context_enhancement: bool = False,
        citation_enhancement: bool = False,
    ):
        self.word_to_senses = word_to_senses
        sense_to_gloss = {row["term"]: row["summary"] for _, row in sense_to_gloss.iterrows()}

        if reduce_dict:
            unique_expansions = set(data["expansion"].unique())
            for word in self.word_to_senses:
                self.word_to_senses[word] = [sense for sense in self.word_to_senses[word] if sense in unique_expansions]

        self.batches = []
        for _, row in tqdm(data.iterrows(), total=len(data)):
            text = row["text"]
            acronym = row["acronym"].lower()
            labels = torch.Tensor([sense == row["expansion"] for sense in self.word_to_senses[acronym]]).float()
            glosses = [sense_to_gloss[sense] for sense in self.word_to_senses[acronym]]
            self.batches.append(
                {
                    "text": text,
                    "examples": eval(row["examples"]) if context_enhancement else eval(row["examples"])[:1],
                    "acronym": acronym,
                    "labels": labels,
                    "glosses": glosses,
                    "expansion": row["expansion"],
                    "paper_titles": eval(row["paper_titles"]) if citation_enhancement else eval(row["paper_titles"])[:1],
                    "paper_id": row["paper_id"],
                }
            )

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        return self.batches[idx]

    @staticmethod
    def collate_fn(batch: Dict[str, any]) -> Dict[str, any]:
        return {
            "text": [item["text"] for item in batch],
            "examples": [item["examples"] for item in batch],
            "acronym": [item["acronym"] for item in batch],
            "labels": torch.cat([item["labels"] for item in batch], dim=0),
            "glosses": [item["glosses"] for item in batch],
            "expansion": [item["expansion"] for item in batch],
            "paper_titles": [item["paper_titles"] for item in batch],
            "paper_id": [item["paper_id"] for item in batch],
        }
