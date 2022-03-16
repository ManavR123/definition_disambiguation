import torch
import torch.nn as nn


class IdentityScoring(nn.Module):
    def __init__(self):
        super(IdentityScoring, self).__init__()

    def forward(self, target, expansion_embeddings):
        return torch.einsum("ijl,il->ij", expansion_embeddings, target)


class LinearScoring(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearScoring, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, target, expansion_embeddings):
        return torch.einsum("ijl,il->ij", expansion_embeddings, target)


class MLPScoring(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(MLPScoring, self).__init__()
        self.model = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, target, expansion_embeddings):
        target = target.unsqueeze(1)
        target = target.repeat(1, expansion_embeddings.shape[1], 1)
        x = torch.cat([expansion_embeddings, target], dim=-1)
        return self.model(x)
