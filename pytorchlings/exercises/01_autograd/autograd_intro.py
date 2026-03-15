"""Exercise 01: autograd intro."""
import torch

def run() -> torch.Tensor:
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    # TODO: build y = x0^2 + 3*x1
    y = x[0]**2 + 3*x[1]
    y.backward()
    # Expected gradient: [4.0, 3.0]
    return x.grad


if __name__ == "__main__":
    print(run())
