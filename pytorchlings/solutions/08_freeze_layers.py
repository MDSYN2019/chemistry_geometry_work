from torch import nn


def freeze_backbone(model: nn.Module) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True
