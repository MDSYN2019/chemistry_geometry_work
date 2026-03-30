"""Exercise 38: choose correct loss usage from logits/probabilities."""
import torch
from torch import nn


def compute_losses(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cross_entropy_loss, bce_with_logits_loss)."""
    ce_targets = targets.long()
    bce_targets = torch.nn.functional.one_hot(targets.long(), num_classes=logits.size(-1)).float()

    # TODO: use nn.CrossEntropyLoss directly on logits + class indices
    ce_loss = nn.CrossEntropyLoss()(torch.softmax(logits, dim=-1), ce_targets)

    # TODO: use nn.BCEWithLogitsLoss on logits + one-hot float targets
    bce_loss = nn.BCELoss()(torch.sigmoid(logits), bce_targets)
    return ce_loss, bce_loss
