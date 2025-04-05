"""
(i) The model

(ii) The loss function

(iii) A minibatch stochastic gradient descent optimizers

(iv) The training function that stitches all of these pieces together


"""

import torch
from d2l import torch as d2l


class LineaRegressionScratch(d2l.Module):
    """
    The linear regression model implemented from scratch
    """

    super().__init__()
    self.save_hyperparameters()
    self.w = torch.normal(sigma, 0, (num_inputs, 1), requires_grad=True)
    self.b = torch.zeros(1, requires_grad=True)


# This decorator for adding in methods is quite useful..
@d2l.add_to_class(LinearRegressionScratch)
def forward(self, X):
    return torch.matmul(X, self.w) + self.b


@d2l.add_to_class(LinearRegressionScratch)
def loss(self, y_hat, y):
    l = (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
    return l.mean()


@d2l.add_to_class(LineaRegressionScratch)
def another_loss(self, y_hat, y):
    pass


# ---

