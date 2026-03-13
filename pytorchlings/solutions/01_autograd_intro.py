import torch


def run() -> torch.Tensor:
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    y = x[0] ** 2 + 3 * x[1]
    y.backward()
    return x.grad
