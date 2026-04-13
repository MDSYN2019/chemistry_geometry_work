"""Exercise 78: 3D CNN starter for protein-ligand voxel scoring.

Goal:
- build a minimal 3D convolutional scorer
- practice volumetric tensor handling and shape debugging
"""

import torch
from torch import nn


class ProteinLigand3DCNN(nn.Module):
    def __init__(self, in_channels: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.head = nn.Linear(32, 1)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        z = self.encoder(grid).flatten(1)

        # TODO: add test-time augmentation via random 3D rotations and average predictions
        return self.head(z).squeeze(-1)


if __name__ == "__main__":
    torch.manual_seed(78)
    batch, channels, size = 4, 8, 20
    voxel_grid = torch.randn(batch, channels, size, size, size)

    model = ProteinLigand3DCNN(in_channels=channels)
    pred = model(voxel_grid)

    assert pred.shape == (batch,)
    print("exercise 78 smoke check passed")
