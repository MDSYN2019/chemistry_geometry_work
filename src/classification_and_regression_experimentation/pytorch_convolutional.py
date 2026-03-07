import torch
from torch import nn
from d2l import torch as d2l

import logging

def corr2d(X, K) -> torch.tensor:
    """
    Compute 2D cross-correlation
    """
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w +1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w ] * K).sum()
    return Y 

if __name__ == "__main__":
    X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    corr2d(X, K)
