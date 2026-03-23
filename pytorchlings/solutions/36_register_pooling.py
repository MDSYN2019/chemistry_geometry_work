import torch
from torch_geometric.graphgym.register import register_pooling
from torch_geometric.utils import scatter


def mean_abs_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Pool by graph using mean of absolute node embeddings."""
    return scatter(x.abs(), batch, dim=0, reduce="mean")


register_pooling("mean_abs", mean_abs_pool)
