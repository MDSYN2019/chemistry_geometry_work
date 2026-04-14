"""Solution 81: ligand-protein cross-attention fusion."""

import torch
from torch import nn


class TokenEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.ff(self.emb(token_ids))


class CrossAttentionScorer(nn.Module):
    def __init__(self, lig_vocab: int, prot_vocab: int, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.lig_enc = TokenEncoder(lig_vocab, d_model)
        self.prot_enc = TokenEncoder(prot_vocab, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.out = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def forward(
        self,
        ligand_ids: torch.Tensor,
        protein_ids: torch.Tensor,
        protein_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        lig = self.lig_enc(ligand_ids)
        prot = self.prot_enc(protein_ids)
        attended, _ = self.cross_attn(
            query=lig,
            key=prot,
            value=prot,
            key_padding_mask=protein_pad_mask,
        )

        pooled = attended.mean(dim=1)
        return self.out(pooled).squeeze(-1)


if __name__ == "__main__":
    torch.manual_seed(81)
    bsz, lig_len, prot_len = 5, 24, 40
    ligand_ids = torch.randint(1, 32, (bsz, lig_len))
    protein_ids = torch.randint(1, 48, (bsz, prot_len))
    protein_ids[0, -5:] = 0
    protein_pad_mask = protein_ids.eq(0)

    model = CrossAttentionScorer(lig_vocab=32, prot_vocab=48, d_model=64, n_heads=4)
    pred = model(ligand_ids, protein_ids, protein_pad_mask=protein_pad_mask)

    assert pred.shape == (bsz,)
    print("solution 81 smoke check passed")
