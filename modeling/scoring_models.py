import torch.nn as nn


class IdentityScoring(nn.Module):
    def __init__(self):
        super(IdentityScoring, self).__init__()

    def forward(self, x):
        return x


class LinearScoring(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearScoring, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)
