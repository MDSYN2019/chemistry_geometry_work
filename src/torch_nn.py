# Neural network modules
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def add_to_class(Class):
    """
    Register functions as methods in created class
    """
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper


    
class MLPScratch(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens, sigma=0.01):
        # Initialize weights with small random values
        self.W1 = nn.Parameter()
        self.b1 = nn.Parameter()


def forward(self, X):
    """
    Forward pass of the neural network - implements the forward computation
    step-by-step
    """
    pass
