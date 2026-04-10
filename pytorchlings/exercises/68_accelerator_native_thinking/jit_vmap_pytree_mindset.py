"""Exercise 68: accelerator-native thinking in a PyTorch-first workflow.

Although JAX has jit/vmap/pytrees as core primitives,
you can practice the same mental model in PyTorch with torch.compile,
torch.func.vmap, and nested parameter structures.
"""

import torch
from torch import nn
from torch.func import functional_call, vmap


def single_example_loss(params: dict[str, torch.Tensor], model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute per-example cross entropy loss via functional_call."""
    logits = functional_call(model, params, (x.unsqueeze(0),))
    return nn.functional.cross_entropy(logits, y.unsqueeze(0))


def batched_loss_with_vmap(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    params = dict(model.named_parameters())

    # TODO: vectorize single_example_loss across batch dimension with vmap
    vmapped = vmap(lambda xb, yb: single_example_loss(params, model, xb, yb))
    losses = vmapped(x, y)
    return losses.mean()


def compile_training_step(model: nn.Module):
    """Return a compiled training-step function."""

    def step(x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        optimizer.zero_grad()
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        return float(loss.item())

    # TODO: tune compile mode/backend options for your environment
    return torch.compile(step)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 3))
    x = torch.randn(16, 10)
    y = torch.randint(0, 3, (16,))

    loss = batched_loss_with_vmap(model, x, y)
    assert loss.ndim == 0

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    step_fn = compile_training_step(model)
    step_loss = step_fn(x, y, optimizer)
    assert step_loss > 0
    print("exercise 68 smoke check passed")
