"""Exercise 24: simple gradient-based node importance."""
import torch


def node_importance(logits: torch.Tensor, x: torch.Tensor, target_node: int, target_class: int) -> torch.Tensor:
    """Return feature gradient magnitudes for one node."""
    # TODO: backprop target logit and return abs gradient for target node
    return torch.zeros_like(x[target_node])
