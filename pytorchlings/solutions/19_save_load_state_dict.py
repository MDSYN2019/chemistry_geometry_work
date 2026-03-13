from pathlib import Path
import torch
from torch import nn


def save_model(model: nn.Module, out_path: Path) -> None:
    torch.save(model.state_dict(), out_path)


def load_model(model: nn.Module, in_path: Path) -> nn.Module:
    state = torch.load(in_path, map_location='cpu')
    model.load_state_dict(state)
    return model
