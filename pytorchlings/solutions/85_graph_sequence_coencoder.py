"""Solution 85: graph + sequence co-encoder."""

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

    def masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = (~mask).unsqueeze(-1)
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)

    def forward(self, x_nodes: torch.Tensor, adj: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        g = self.encode_graph(x_nodes, adj)
        s = self.token_emb(token_ids)
        seq_pad = token_ids.eq(0)

        s_ctx, _ = self.seq_to_graph(query=s, key=g, value=g)
        g_ctx, _ = self.graph_to_seq(query=g, key=s, value=s, key_padding_mask=seq_pad)

        s_pool = self.masked_mean(s_ctx, seq_pad)
        g_pool = g_ctx.mean(dim=1)
        fused = torch.cat([s_pool, g_pool], dim=-1)
        return self.head(fused).squeeze(-1)


if __name__ == "__main__":
    torch.manual_seed(85)
    bsz, n_nodes, node_dim, seq_len = 4, 15, 20, 18
    x_nodes = torch.randn(bsz, n_nodes, node_dim)
    token_ids = torch.randint(1, 25, (bsz, seq_len))
    token_ids[0, -4:] = 0

    a = torch.randint(0, 2, (bsz, n_nodes, n_nodes), dtype=torch.float32)
    adj = torch.triu(a, diagonal=1)
    adj = adj + adj.transpose(-1, -2)
    adj = (adj + torch.eye(n_nodes).unsqueeze(0)).clamp(max=1.0)

    model = GraphSequenceCoEncoder(node_dim=node_dim, vocab_size=25, hidden=64)
    y = model(x_nodes, adj, token_ids)
    assert y.shape == (bsz,)
    print("solution 85 smoke check passed")
