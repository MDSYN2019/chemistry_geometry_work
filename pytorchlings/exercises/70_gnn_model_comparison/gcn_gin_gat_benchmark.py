"""Exercise 70: compare GCN, GIN, and GAT on one molecular task.

What to implement:
1) shared training/evaluation loops
2) consistent splits and seeds
3) one ablation on node/edge feature choices
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, GINConv, global_mean_pool


class GCNModel(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        return self.head(global_mean_pool(h, batch))


class GINModel(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        mlp1 = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        mlp2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.conv1 = GINConv(mlp1)
        self.conv2 = GINConv(mlp2)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        return self.head(global_mean_pool(h, batch))


class GATModel(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, heads: int = 4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=heads)
        self.conv2 = GATConv(hidden * heads, hidden, heads=1)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        return self.head(global_mean_pool(h, batch))


@dataclass
class RunResult:
    model_name: str
    val_metric: float


def train_eval_stub(model: nn.Module) -> float:
    """TODO: replace with full train/val loop using scaffold split and identical protocol."""
    n_params = sum(p.numel() for p in model.parameters())
    return float(torch.log(torch.tensor(n_params, dtype=torch.float32)).item())


if __name__ == "__main__":
    in_dim, hidden, out_dim = 16, 32, 1
    models = {
        "gcn": GCNModel(in_dim, hidden, out_dim),
        "gin": GINModel(in_dim, hidden, out_dim),
        "gat": GATModel(in_dim, hidden, out_dim),
    }

    results = [RunResult(name, train_eval_stub(m)) for name, m in models.items()]
    assert len(results) == 3
    print(results)
