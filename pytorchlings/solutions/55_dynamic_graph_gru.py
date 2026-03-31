"""Solution 55: dynamic GNN over temporal snapshots."""
import torch
from torch import nn
from torch_geometric.nn import GCNConv


class DynamicGCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.gcn = GCNConv(in_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)

    def forward(self, x_seq: list[torch.Tensor], edge_seq: list[torch.Tensor]) -> torch.Tensor:
        # encode each time step with shared GCN, then pass node trajectories into GRU.
        step_embeds = [self.gcn(x_t, e_t) for x_t, e_t in zip(x_seq, edge_seq)]
        stacked = torch.stack(step_embeds, dim=1)  # [num_nodes, time, hidden]
        out, _ = self.gru(stacked)
        return out[:, -1, :]
