"""Exercise 82: multimodal architecture with uncertainty head.

Builds on:
- Exercise 79 (fingerprint + graph fusion)
- Exercise 81 (cross-attention fusion)

Goal:
- combine three modality encoders (fingerprint, graph, sequence)
- output mean + log-variance for heteroscedastic regression
"""

import torch
from torch import nn


class MultiModalPropertyModel(nn.Module):
    def __init__(self, fp_dim: int, node_dim: int, vocab_size: int, hidden: int = 96):
        super().__init__()
        self.fp_encoder = nn.Sequential(nn.Linear(fp_dim, hidden), nn.ReLU())
        self.node_proj = nn.Linear(node_dim, hidden)
        self.token_emb = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.token_attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)

        self.trunk = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.logvar_head = nn.Linear(hidden, 1)

    def encode_graph(self, x_nodes: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neigh_mean = (adj @ x_nodes) / deg
        h = torch.relu(self.node_proj(0.5 * (x_nodes + neigh_mean)))
        return h.mean(dim=1)

    def encode_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        tok = self.token_emb(token_ids)
        attended, _ = self.token_attn(tok, tok, tok)

        # TODO: add attention mask support so padded positions are ignored
        return attended.mean(dim=1)

    def forward(
        self,
        x_fp: torch.Tensor,
        x_nodes: torch.Tensor,
        adj: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fp_repr = self.fp_encoder(x_fp)
        graph_repr = self.encode_graph(x_nodes, adj)
        token_repr = self.encode_tokens(token_ids)

        fused = torch.cat([fp_repr, graph_repr, token_repr], dim=-1)
        h = self.trunk(fused)
        mean = self.mean_head(h).squeeze(-1)
        logvar = self.logvar_head(h).squeeze(-1)
        return mean, logvar


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    inv_var = torch.exp(-logvar)
    return 0.5 * torch.mean(logvar + (target - mean) ** 2 * inv_var)


if __name__ == "__main__":
    torch.manual_seed(82)
    batch, fp_dim, n_nodes, node_dim, seq_len = 4, 256, 14, 32, 22
    x_fp = torch.randn(batch, fp_dim)
    x_nodes = torch.randn(batch, n_nodes, node_dim)
    token_ids = torch.randint(1, 30, (batch, seq_len))

    a = torch.randint(0, 2, (batch, n_nodes, n_nodes), dtype=torch.float32)
    adj = torch.triu(a, diagonal=1)
    adj = adj + adj.transpose(-1, -2)
    adj = (adj + torch.eye(n_nodes).unsqueeze(0)).clamp(max=1.0)

    target = torch.randn(batch)
    model = MultiModalPropertyModel(fp_dim=fp_dim, node_dim=node_dim, vocab_size=30, hidden=64)
    mean, logvar = model(x_fp, x_nodes, adj, token_ids)
    loss = gaussian_nll(mean, logvar, target)

    assert mean.shape == (batch,)
    assert logvar.shape == (batch,)
    assert torch.isfinite(loss)
    print("exercise 82 smoke check passed")
