"""Exercise 86: Mixture-of-Experts multimodal router.

Builds on:
- Exercise 82 (multimodal fusion + uncertainty)
- Exercise 83 (residual MLP backbones)
- Exercise 85 (co-encoding modalities)

Goal:
- route fused representations through multiple expert heads
- learn soft expert assignments and aggregate predictions
"""

import torch
from torch import nn


class MoEMultimodalRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, n_experts: int = 3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.router = nn.Linear(hidden, n_experts)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        gate_logits = self.router(h)
        weights = torch.softmax(gate_logits, dim=-1)

        expert_preds = torch.cat([expert(h) for expert in self.experts], dim=-1)

        # TODO: add entropy regularization term for stable expert utilization
        pred = (weights * expert_preds).sum(dim=-1)
        return pred, weights


if __name__ == "__main__":
    torch.manual_seed(86)
    x = torch.randn(12, 192)
    model = MoEMultimodalRegressor(in_dim=192, hidden=96, n_experts=4)
    pred, weights = model(x)

    assert pred.shape == (12,)
    assert weights.shape == (12, 4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(12), atol=1e-5)
    print("exercise 86 smoke check passed")
