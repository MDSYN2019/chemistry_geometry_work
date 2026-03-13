import torch
from torch_geometric.utils import negative_sampling


def sample_negative(edge_index: torch.Tensor, num_nodes: int, num_neg: int) -> torch.Tensor:
    return negative_sampling(edge_index=edge_index, num_nodes=num_nodes, num_neg_samples=num_neg)
