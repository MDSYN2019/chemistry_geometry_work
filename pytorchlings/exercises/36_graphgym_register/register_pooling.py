"""Exercise 36: register a custom GraphGym global pooling op."""
import torch
from torch_geometric.graphgym.register import register_pooling


def mean_abs_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Pool by graph using mean of absolute node embeddings."""
    # TODO: return scatter mean(abs(x)) grouped by batch
    # Hint: use torch_geometric.utils.scatter with reduce="mean"
    return x


# TODO: register the function above under the name "mean_abs"
register_pooling("todo_mean_abs", mean_abs_pool)
