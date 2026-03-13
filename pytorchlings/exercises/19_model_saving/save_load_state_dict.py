"""Exercise 19: save/load model state dict."""
from pathlib import Path
import torch
from torch import nn


def save_model(model: nn.Module, out_path: Path) -> None:
    # TODO: save model state_dict to out_path
    pass


def load_model(model: nn.Module, in_path: Path) -> nn.Module:
    # TODO: load state_dict from in_path into model and return it
    return model
