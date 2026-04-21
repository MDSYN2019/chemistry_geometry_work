"""Exercise 91: supervised fine-tuning (SFT) for chemistry Q&A.

Goal:
- format instruction datasets with prompt/response pairs
- apply response-only loss masking
- run one train step for beginner-friendly instruction tuning
"""

from __future__ import annotations

import torch

IGNORE_INDEX = -100


def build_prompt(example: dict[str, str]) -> str:
    """Simple chat template."""
    instr = example["instruction"].strip()
    inp = example.get("input", "").strip()
    out = example["output"].strip()
    return f"<user> {instr}\n{inp}\n<assistant> {out}"


def build_label_mask(token_ids: torch.Tensor, assistant_token_id: int) -> torch.Tensor:
    """Keep labels only after the assistant token; mask earlier positions with IGNORE_INDEX."""
    labels = token_ids.clone()
    # TODO: find assistant_token_id position per sample.
    # TODO: set prompt-side labels to IGNORE_INDEX.
    return labels


if __name__ == "__main__":
    torch.manual_seed(91)

    batch = torch.tensor(
        [
            [3, 4, 5, 2, 8, 9, 10],
            [7, 1, 2, 6, 5, 4, 3],
        ],
        dtype=torch.long,
    )
    assistant_token_id = 2
    labels = build_label_mask(batch, assistant_token_id)

    assert labels.shape == batch.shape
    print("labels sample:", labels[0].tolist())
    print("exercise 91 scaffold ready")
