import random
import torch
from d2l import torch as d2l


class SyntheticRegressionData(d2l.DataModule):  # @save
    """Synthetic data for linear regression."""

    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, len(w))  # get randomly distributed dat aof n and len(w)
        noise = torch.randn(n, 1) * noise
        self.y = torch.matmul(self.X, w.reshape((-1, 1))) + b + noise


@d2l.add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    if train:
        indices = list(range(0, self.num_train))
        # The examples are read in random order
        random.shuffle(indices)
    else:
        indices = list(range(self.num_train, self.num_train + self.num_val))

    for i in range(0, len(indices), self.batch_size):
        batch_indices = torch.tensor(indices[i : i + self.batch_size])

    for i in range(0, len(indices), self.batch_size):
        batch_indices = torch.tensor(indices[i : i + self.batch_size])
        yield self.X[batch_indices], self.y[batch_indices]


d2l.add_to_class(d2l.DataModule)


def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    """ """
    tensors = tuple(
        a[indices] for a in tensors
    )  # not sure what this dataset looks like
    dataset = torch.utils.data.TensorDataset(*tensors)
    return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=train)


# TODO


if __name__ == "__main__":
    data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
    print(data)
