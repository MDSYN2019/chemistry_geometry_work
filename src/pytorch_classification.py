import torch
from torch import nn
from tqdm import tqdm

from sklearn.datasets import make_circles

n_samples = 100
X, y = make_circles(n_samples, noise = 0.03, random_state = 42)


