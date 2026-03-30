import torch
from torch import nn


def compute_losses(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ce_targets = targets.long()
    bce_targets = torch.nn.functional.one_hot(targets.long(), num_classes=logits.size(-1)).float()

    ce_loss = nn.CrossEntropyLoss()(logits, ce_targets)
    bce_loss = nn.BCEWithLogitsLoss()(logits, bce_targets)
    return ce_loss, bce_loss
