import torch
from torch_geometric.data import HeteroData


def build_hetero() -> HeteroData:
    data = HeteroData()
    data['author'].x = torch.randn(4, 8)
    data['paper'].x = torch.randn(6, 8)
    data['author', 'writes', 'paper'].edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    data['paper', 'written_by', 'author'].edge_index = data['author', 'writes', 'paper'].edge_index.flip(0)
    return data
