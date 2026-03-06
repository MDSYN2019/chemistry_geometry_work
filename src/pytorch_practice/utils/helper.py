import torch
from timeit import default_timer as timer


def print_train_time(start: float, end: float, device: torch.device = None):
    """
    Prints difference between start and end time
    """
    total_time = end - start
    return total_time
