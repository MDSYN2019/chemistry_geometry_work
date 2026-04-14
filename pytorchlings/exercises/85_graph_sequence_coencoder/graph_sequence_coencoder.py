"""Exercise 85: graph + sequence co-encoder.

Builds on:
- Exercise 76 (graph message passing)
- Exercise 77 (sequence encoding)
- Exercise 81 (cross-attention)

Goal:
- encode graph nodes and sequence tokens in parallel
- perform bidirectional cross-attention before final fusion
"""

import torch
from torch import nn


class GraphSequenceCoEncoder(nn.Module):
    def __init__(self, node_dim: int, vocab_size: int, hidden: int = 64):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden)
        self.token_emb = nn.Embedding(vocab_size, hidden, padding_idx=0)

        self.seq_to_graph = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.graph_to_seq = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)

        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def encode_graph(self, x_nodes: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neigh = (adj @ x_nodes) / deg
        return torch.relu(self.node_proj(0.5 * (x_nodes + neigh)))

    def forward(self, x_nodes: torch.Tensor, adj: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        g = self.encode_graph(x_nodes, adj)
        s = self.token_emb(token_ids)

        s_ctx, _ = self.seq_to_graph(query=s, key=g, value=g)
        g_ctx, _ = self.graph_to_seq(query=g, key=s, value=s)

        # TODO: add masks and weighted readouts per modality
        fused = torch.cat([s_ctx.mean(dim=1), g_ctx.mean(dim=1)], dim=-1)
        return self.head(fused).squeeze(-1)


if __name__ == "__main__":
    torch.manual_seed(85)
    bsz, n_nodes, node_dim, seq_len = 4, 15, 20, 18
    x_nodes = torch.randn(bsz, n_nodes, node_dim)
    token_ids = torch.randint(1, 25, (bsz, seq_len))

    a = torch.randint(0, 2, (bsz, n_nodes, n_nodes), dtype=torch.float32)
    adj = torch.triu(a, diagonal=1)
    adj = adj + adj.transpose(-1, -2)
    adj = (adj + torch.eye(n_nodes).unsqueeze(0)).clamp(max=1.0)

    model = GraphSequenceCoEncoder(node_dim=node_dim, vocab_size=25, hidden=64)
    y = model(x_nodes, adj, token_ids)
    assert y.shape == (bsz,)
    print("exercise 85 smoke check passed")
