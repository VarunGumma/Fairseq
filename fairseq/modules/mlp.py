import torch.nn as nn
from fairseq import utils


# a standard MLP with a single hidden layer
class MLP(nn.Module):
    def __init__(
        self, in_features, intermediate_features, activation_fn="relu", bias=True
    ):
        super().__init__()
        self.in_features = in_features
        self.intermediate_features = intermediate_features
        self.act_fn = utils.get_activation_fn(activation_fn)
        self.fc1 = nn.Linear(self.in_features, self.intermediate_features, bias=bias)
        self.fc2 = nn.Linear(self.intermediate_features, self.in_features, bias=bias)

    def forward(self, x):
        return self.fc2(self.act_fn(self.fc1(x)))


# Gated Linear Unit as proposed by Noam Shazeer in `GLU Variants Improve Transformer`
class GLU(nn.Module):
    def __init__(
        self, in_features, intermediate_features, activation_fn="silu", bias=False
    ):
        super().__init__()
        self.in_features = in_features
        self.intermediate_features = intermediate_features
        self.act_fn = utils.get_activation_fn(activation_fn)
        self.fc1 = nn.Linear(self.in_features, self.intermediate_features, bias=bias)
        self.fc2 = nn.Linear(self.in_features, self.intermediate_features, bias=bias)
        self.fc3 = nn.Linear(self.intermediate_features, self.in_features, bias=bias)

    def forward(self, x):
        return self.fc3(self.act_fn(self.fc2(x)) * self.fc1(x))
