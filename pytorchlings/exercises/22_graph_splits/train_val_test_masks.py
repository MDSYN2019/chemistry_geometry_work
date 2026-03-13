"""Exercise 22: node train/val/test mask creation."""
import torch


def make_masks(num_nodes: int):
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    # TODO: assign 60% train, 20% val, 20% test (contiguous index slices)
    return train_mask, val_mask, test_mask
