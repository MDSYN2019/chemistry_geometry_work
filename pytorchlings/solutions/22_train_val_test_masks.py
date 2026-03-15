import torch


def make_masks(num_nodes: int):
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    n_train = int(0.6 * num_nodes)
    n_val = int(0.2 * num_nodes)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    return train_mask, val_mask, test_mask
