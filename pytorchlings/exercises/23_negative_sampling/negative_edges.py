"""Exercise 23: negative sampling for link prediction."""
import torch
from torch_geometric.utils import negative_sampling


def sample_negative(edge_index: torch.Tensor, num_nodes: int, num_neg: int) -> torch.Tensor:
    # TODO: call negative_sampling with num_neg_samples=num_neg
    return edge_index
