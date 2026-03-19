"""
A small pytorch toy example that mirrors the structure


- Build a handcrafted geometric feature vector g(s_j) for each probe
- apply an MLP to each probe feature
- pool across probes
- also pool raw features first, then pass through another MLP
- concatenate both branches
- use a final head for prediction

"""


import torch
import torch.nn as nn

class ProbeSetEncoder(nn.Module):
    def __init__(
            self,
            probe_feat_dim: int = 7,
            hidden_dim: int = 32,
            pooled_dim: int = 32,
            out_dim: int = 1,
            pooling: str = "mean"
    ):
        super().__init__()
        if pooling not in {"mean", "max", "sum"}:
            raise ValueError("pooling must be one of mean max or sum")
        
        self.pooling = pooling# by default this would be mean

        # Branch 1 : MLP applied  to each probe feature g(s_j)
        self.probe_mlp = nn.Sequential(
            nn.Linear(probe_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pooled_dim),
            nn.ReLU()
        )


        # Branch 2: pool raw probe features first, then apply MLP
        
        self.global_mlp = nn.Sequential(
            nn.Linear(probe_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pooled_dim),
            nn.ReLU()
        )

        # Final prediction head after concatenation

        self.head = nn.Sequential(
            nn.Linear(2 * pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def pool(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        x: tensor of shape (..., num_items, feat_dim)
        returns pooled tensor over `dim`
        """
        if self.pooling == "mean":
            return x.mean(dim=dim)
        elif self.pooling == "sum":
            return x.sum(dim=dim)
        elif self.pooling == "max":
            return x.max(dim=dim).values
        else:
            raise RuntimeError("Unexpected pooling type")

    
    def forward(self, probe_features: torch.Tensor) -> torch.Tensor:
        """
        """
        per_probe_embed = self.probe_mlp(probe_features)
        # (batch_size, num_probes, pooled_dim)

        branch1 = self.pool(per_probe_embed, dim = 1)
        pooled_raw = self.pool(probe_features, dim = 1)
        branch2 = self.global_mlp(pooled_raw)

        combined = torch.cat([branch1, branch2], dim = -1)
        prediction = self.head(combined)
        return prediction



        
def make_toy_probe_features(batch_size: int = 4, num_probes: int = 10) -> torch.Tensor:
    """
    Create fake geometric features g(s_j) of length 7 for each probe.

    Interpreted as:
    [||x_jj1||, ||x_jj2||, angle1_cos,num_probes = 10
     ||x_j,center||, ||x_j,protein||, ||x_protein,center||, angle2_cos]
    """
    distances = torch.rand(batch_size, num_probes, 5) * 10.0
    angles = torch.rand(batch_size, num_probes, 2) * 2.0 - 1.0  # cosine-like values in [-1, 1]

    # Put angles in slots 2 and 6
    feat = torch.empty(batch_size, num_probes, 7)
    feat[:, :, 0] = distances[:, :, 0]
    feat[:, :, 1] = distances[:, :, 1]
    feat[:, :, 2] = angles[:, :, 0]
    feat[:, :, 3] = distances[:, :, 2]
    feat[:, :, 4] = distances[:, :, 3]
    feat[:, :, 5] = distances[:, :, 4]
    feat[:, :, 6] = angles[:, :, 1]
    return feat


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 4
    num_probes = 10

    x = make_toy_probe_features(batch_size, num_probes)

    
    
    model = ProbeSetEncoder(
        probe_feat_dim=7,
        hidden_dim=32,
        pooled_dim=32,
        out_dim=1,
        pooling="mean",
    )
    
    y = model(x)
    print("Output shape:", y.shape)  # (4, 1)
    print("Predictions:\n", y)
