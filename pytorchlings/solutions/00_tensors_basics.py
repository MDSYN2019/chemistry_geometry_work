"""Solution 00."""
import torch


def run() -> torch.Tensor:
    x = torch.arange(0, 6, dtype=torch.float32).reshape(2, 3)
    row_sum = x.sum(dim=1)
    return row_sum


if __name__ == "__main__":
    print(run())
