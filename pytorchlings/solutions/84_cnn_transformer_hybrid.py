"""Solution 84: CNN -> Transformer hybrid encoder."""

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

    def sinusoidal_2d_pos(self, h: int, w: int, d_model: int, device: torch.device) -> torch.Tensor:
        y = torch.arange(h, device=device).float().unsqueeze(1)
        x = torch.arange(w, device=device).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model // 2, 2, device=device).float() * (-torch.log(torch.tensor(10000.0, device=device)) / (d_model // 2)))

        py = torch.zeros(h, d_model // 2, device=device)
        px = torch.zeros(w, d_model // 2, device=device)
        py[:, 0::2] = torch.sin(y * div)
        py[:, 1::2] = torch.cos(y * div)
        px[:, 0::2] = torch.sin(x * div)
        px[:, 1::2] = torch.cos(x * div)

        pos = torch.zeros(h, w, d_model, device=device)
        pos[:, :, : d_model // 2] = py[:, None, :]
        pos[:, :, d_model // 2 :] = px[None, :, :]
        return pos.reshape(1, h * w, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.cnn(x)
        b, c, h, w = fmap.shape
        tokens = fmap.permute(0, 2, 3, 1).reshape(b, h * w, c)
        tokens = tokens + self.sinusoidal_2d_pos(h, w, c, tokens.device)
        z = self.encoder(tokens)
        pooled = z.mean(dim=1)
        return self.head(pooled).squeeze(-1)


if __name__ == "__main__":
    torch.manual_seed(84)
    x = torch.randn(6, 3, 32, 32)
    model = CNNTransformerHybrid(in_channels=3, d_model=64, n_heads=4)
    y = model(x)
    assert y.shape == (6,)
    print("solution 84 smoke check passed")
