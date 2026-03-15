"""Exercise 16: learning-rate scheduler step."""
import torch


def run_scheduler_step(optim: torch.optim.Optimizer) -> float:
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)
    # TODO: step scheduler once and return new lr
    return float(optim.param_groups[0]["lr"])
