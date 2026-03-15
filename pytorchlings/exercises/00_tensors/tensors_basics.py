"""Exercise 00: tensor creation and basic ops."""
import torch


def run() -> torch.Tensor:
    # TODO: create a float tensor shaped (2, 3) with values 0..5
    x = torch.empty(2, 3)

    # TODO: compute row-wise sum (shape should be (2,))
    row_sum = x.sum(dim=1)
    return row_sum


if __name__ == "__main__":
    out = run()
    print(out)
