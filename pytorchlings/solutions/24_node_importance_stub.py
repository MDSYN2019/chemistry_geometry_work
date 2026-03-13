import torch


def node_importance(logits: torch.Tensor, x: torch.Tensor, target_node: int, target_class: int) -> torch.Tensor:
    target = logits[target_node, target_class]
    target.backward(retain_graph=True)
    return x.grad[target_node].abs()
