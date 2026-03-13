"""Exercise 20: construct hetero graph data."""
import torch
from torch_geometric.data import HeteroData


def build_hetero() -> HeteroData:
    data = HeteroData()
    data['author'].x = torch.randn(4, 8)
    data['paper'].x = torch.randn(6, 8)
    # TODO: add edges author->paper and reverse paper->author
    data['author', 'writes', 'paper'].edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    return data
