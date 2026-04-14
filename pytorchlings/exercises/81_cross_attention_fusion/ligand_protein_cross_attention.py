"""Exercise 81: ligand-protein cross-attention fusion.

Builds on:
- Exercise 77 (sequence encoders)
- Exercise 78 (protein-ligand architecture context)

Goal:
- encode ligand tokens and protein-pocket tokens
- use cross-attention to condition ligand representation on protein context
"""

import torch
from torch import nn


class TokenEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.ff(self.emb(token_ids))


class CrossAttentionScorer(nn.Module):
    def __init__(self, lig_vocab: int, prot_vocab: int, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.lig_enc = TokenEncoder(lig_vocab, d_model)
        self.prot_enc = TokenEncoder(prot_vocab, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.out = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, ligand_ids: torch.Tensor, protein_ids: torch.Tensor) -> torch.Tensor:
        lig = self.lig_enc(ligand_ids)
        prot = self.prot_enc(protein_ids)

        # ligand queries protein context
        attended, _ = self.cross_attn(query=lig, key=prot, value=prot)

        # TODO: add padding masks for variable-length batches
        pooled = attended.mean(dim=1)
        return self.out(pooled).squeeze(-1)


if __name__ == "__main__":
    torch.manual_seed(81)
    bsz, lig_len, prot_len = 5, 24, 40
    ligand_ids = torch.randint(1, 32, (bsz, lig_len))
    protein_ids = torch.randint(1, 48, (bsz, prot_len))

    model = CrossAttentionScorer(lig_vocab=32, prot_vocab=48, d_model=64, n_heads=4)
    pred = model(ligand_ids, protein_ids)

    assert pred.shape == (bsz,)
    print("exercise 81 smoke check passed")
