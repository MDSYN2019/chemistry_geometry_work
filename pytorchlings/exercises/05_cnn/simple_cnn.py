"""Exercise 05: CNN forward pass."""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root = "data",
    train = True,
    download = True,
    transform = transform,
)

test_dataset = datasets.MNIST(
    root = "data",
    train = False,
    download = True,
    transform = transform
)
    
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # TODO: fix input dimension for flattened 14x14 feature map
        #self.classifier = nn.Linear(8 * 28 * 28, 10)
        self.classifier = nn.Linear(8 * 14 * 14, 10)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = h.flatten(start_dim=1)
        return self.classifier(h)
