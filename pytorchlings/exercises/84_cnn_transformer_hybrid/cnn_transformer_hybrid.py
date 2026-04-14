"""Exercise 84: CNN -> Transformer hybrid encoder.

Builds on:
- Exercise 05 (CNN feature extraction)
- Exercise 18 (transformer block)

Goal:
- turn 2D feature maps into token sequences
- apply transformer encoding before global pooling
"""

import torch
from torch import nn


class CNNTransformerHybrid(nn.Module):
    def __init__(self, in_channels: int = 3, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True),
            num_layers=2,
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.cnn(x)  # [B, C, H, W]
        b, c, h, w = fmap.shape
        tokens = fmap.permute(0, 2, 3, 1).reshape(b, h * w, c)

        # TODO: add positional encodings so the transformer keeps spatial context
        z = self.encoder(tokens)
        pooled = z.mean(dim=1)
        return self.head(pooled).squeeze(-1)


if __name__ == "__main__":
    torch.manual_seed(84)
    x = torch.randn(6, 3, 32, 32)
    model = CNNTransformerHybrid(in_channels=3, d_model=64, n_heads=4)
    y = model(x)
    assert y.shape == (6,)
    print("exercise 84 smoke check passed")
