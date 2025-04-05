import torch
from d2l import torch as d2l
import typing as t
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""
-> Stochastic Gradient Descent (compute_sgd):

Updates parameters one observation at a time.

-> Mini-Batch Gradient Descent (compute_batch_sgd):
.
Divides the dataset into mini-batches, computes the gradients for each batch, and
updates parameters using the averaged gradients

---

Recall that the synthetic regression that we generated does not provide a validation
dataset. In most cases, however, we will want a validation dataset to measure
our model quality.

"""


# SGD loop
def compute_sgd(X, Y, w, b, learning_rate) -> t.Tuple[int]:
    """
    Simple implementation of a stochastic gradient descent
    """
    for epoch in range(10):
        for x_i, y_i in zip(X, Y):
            y_pred = w * x_i + b

            # Compute the gradients
            dw = -(y_i - y_pred) * x_i
            db = -(y_i - y_pred)

            # Update parameters
            w -= learning_rate * dw
            b -= learning_rate * db

        # logging.info(f"Epoch {epoch + 1}: w = {w: 4f}, b = {b: 4f} for normal sgd")

    return w, b


def compute_batch_sgd(X, Y, w, b, learning_rate, batch_size) -> t.Tuple[int]:
    """

    Implementation of mini-batch stochastic gradient descent (SGD)
    --------------------------------------------------------------

    In the most basic form, in each iteration t, we first randomly sample a minibatch
    Bt consisting of a fixed number B of training example. We then finally compute the derivative of the average loss
    on in the minibatch with respect to the model parameteres


    """
    n_samples = len(X)

    # call on compute_sgd again
    for epoch in range(100):
        # shuffles the data
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        # regenerate a shuffles index
        X, Y = X[indices], Y[indices]

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            X_batch = X[start:end]
            y_batch = Y[start:end]

            dw, db = 0.0, 0.0  # Compute the gradients for the batch

            for x_i, y_i in zip(X_batch, y_batch):
                y_pred = w * x_i + b
                dw += -(y_i - y_pred) * x_i
                db += -(y_i - y_pred)

            dw /= len(X_batch)
            db /= len(X_batch)

            w -= learning_rate * dw
            b -= learning_rate * db

        logging.info(
            f"Epoch {epoch + 1}: w = {w: 4f}, b = {b: 4f} for batch normal sgd"
        )

    return w, b


# d2l based implementation
class SGD(d2l.HyperParameters):
    """
    Minibatch stochastic gradient descent
    """

    def __init__(self, params, lr):
        self.save_parameters()

    def step(self):
        for param in self.params:
            param.data -= (
                self.lr * param.grad
            )  # learning rate I think we are computing both for the w and bias

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


if __name__ == "__main__":
    # Training data
    X = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    Y = np.array([2, 4, 6, 8, 10, 2, 4, 6, 8, 10, 2, 4, 6, 8, 10])
    # Initial Parameters
    w = 0.0
    b = 0.0
    learning_rate = 0.01
    # w, b = compute_sgd(X, Y, w, b, learning_rate)
    w, b = compute_batch_sgd(X, Y, w, b, learning_rate, 5)
    print(f"we have {w}, {b}")
