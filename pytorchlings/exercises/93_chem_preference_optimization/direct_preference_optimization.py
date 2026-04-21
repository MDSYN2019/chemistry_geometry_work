"""Exercise 93: Direct Preference Optimization (DPO) mechanics.

Goal:
- understand chosen vs rejected response pairs
- implement DPO-style objective from log-probabilities
- connect post-training alignment to scientific assistant behavior
"""

from __future__ import annotations

import torch


def dpo_loss(
    logp_policy_chosen: torch.Tensor,
    logp_policy_rejected: torch.Tensor,
    logp_ref_chosen: torch.Tensor,
    logp_ref_rejected: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Compute scalar DPO loss over a batch."""
    # TODO: implement:
    # logits = beta * ((logp_policy_chosen - logp_policy_rejected) - (logp_ref_chosen - logp_ref_rejected))
    # loss = -log(sigmoid(logits)).mean()
    return torch.tensor(0.0)


if __name__ == "__main__":
    torch.manual_seed(93)

    bsz = 4
    policy_c = torch.randn(bsz)
    policy_r = torch.randn(bsz)
    ref_c = torch.randn(bsz)
    ref_r = torch.randn(bsz)

    loss = dpo_loss(policy_c, policy_r, ref_c, ref_r, beta=0.2)
    print("dpo loss:", float(loss.item()))
    print("exercise 93 scaffold ready")
