import torch
from torch import nn

"""
A classification problem involves whether something is one thing or another

For example, you might want to:

- Binary classification
- Multi-class classification
- Multi-label classification
"""


from sklearn.datasets import make_circles

# make 1000 samples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples, noise = 0.03, random_state = 42)

