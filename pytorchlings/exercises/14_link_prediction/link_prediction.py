"""Exercise 14: link prediction decoder."""
import torch
from torch import nn
from torch_geometric.nn import SAGEConv


class LinkPredictor(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.scorer = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index).relu()
        return self.conv2(h, edge_index)

    def decode(self, z: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        # edge_pairs shape: [2, E]
        src = z[edge_pairs[0]]
        dst = z[edge_pairs[1]]
        # TODO: pass concatenated node embeddings through scorer
        return (src * dst).sum(dim=-1)
