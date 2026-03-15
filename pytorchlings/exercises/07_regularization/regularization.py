"""Exercise 07: regularization."""
from torch import nn


def build_model() -> nn.Module:
    # TODO: insert dropout between layers
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )

# TODO: when creating Adam optimizer, add weight_decay=1e-4
