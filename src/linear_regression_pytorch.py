import torch
from torch import nn


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype = torch.float))
        self.bias = nn.Parameter(torch.randn(1, dtype = torch.float), requires_grad = True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward defines the computation in the model 
        """
        return self.weights * x + self.bias

    
