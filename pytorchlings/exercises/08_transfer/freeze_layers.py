"""Exercise 08: transfer learning freeze/unfreeze."""
from torch import nn


def freeze_backbone(model: nn.Module) -> None:
    # Convention: model has .backbone and .head modules.
    for p in model.backbone.parameters():
        # TODO: set to False
        p.requires_grad = True

    for p in model.head.parameters():
        # TODO: set to True explicitly
        p.requires_grad = p.requires_grad
