"""Exercise 00: tensor creation and basic ops."""
import torch


def run() -> torch.Tensor:
    # TODO: create a float tensor shaped (2, 3) with values 0..5
    #x = torch.empty(2, 3, dtype = torch.float32)
    x_true = torch.arange(0,6, dtype = torch.float32).reshape(2,3)
    print(x_true)
    # TODO: compute row-wise sum (shape should be (2,))
    x_true_row_sum = x_true.sum(dim=1)
    #row_sum = x.sum(dim=0)
    return x_true_row_sum

if __name__ == "__main__":
    out = run()
    print(out)
