from abc import ABC

import torch
from torch import nn


class WSDModel(ABC, nn.Module):
    def __init__(self):
        super(WSDModel, self).__init__()

    def forward(self):
        raise NotImplementedError

    def create_input(self):
        raise NotImplementedError

    def get_scores(self) -> torch.Tensor:
        raise NotImplementedError

    def step(self, batch, device) -> torch.Tensor:
        raise NotImplementedError
