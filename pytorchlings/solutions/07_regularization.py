from torch import nn


def build_model() -> nn.Module:
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 2))
