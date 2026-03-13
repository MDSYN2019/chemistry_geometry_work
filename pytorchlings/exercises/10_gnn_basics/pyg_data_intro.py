"""Exercise 10: build a minimal PyG Data object."""

# pip install torch-geometric
import torch
from torch_geometric.data import Data


def build_graph() -> Data:
    # 3 nodes with 2 features each
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    # TODO: make an undirected chain 0-1-2
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)
