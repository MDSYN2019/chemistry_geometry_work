
import time
import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import torchvision
from torchvision import transforms, datasets
import logging

class FashionMNISTDataModule:
    def __init__(self, batch_size = 64, resize = (28, 28), root = './data'):
        self.batch_size = batch_size
        self.resize = resize
        self.root = root
        self._prepare_transforms()
        self._prepare_datasets()

    def _prepare_transforms(self):
        """
        Resize to the resize dimensions, and then convert this to a tensor
        """
        self.transform  = transforms.Compose([
            transforms.Resize(self.resize), 
            transforms.ToTensor()
        ])

    def _prepare_datasets(self):
        self.train_dataset = datasets.FashionMNIST(
            root = self.root,
            train = True, # ensure that it is a training dataset 
            transform = self.transform,
            download = True
        )

        self.val_dataset = datasets.FashionMNIST(
            root = self.root,
            train = False,
            transform = self.transform,
            download = True
        )

    def get_dataloaders(self):
        train_loader = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)
        val_loader = DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle = False)
        return train_loader, val_loader
    
    
class BasicClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features = 2, out_features = 5)
        
        self.layer_2 = nn.Linear(in_features = 5, out_features = 1)
                                     
    def forward(self, x):
        return self.layer_2(self.layer_1(x))
    

if __name__ == "__main__":

    """
    Setup loss function and optimizer

    For a regression problem, you might use mean absolute error (MAE) loss

    For a binary classification problem, you'll often use binary cross entropy as the loss function

    However, the same optimizer function can often be used across different problem spaces
    
    """
    data_module = FashionMNISTDataModule(batch_size = 32, resize = (28, 28))
    train_loader, val_loader = data_module.get_dataloaders()
    print(train_loader, val_loader)

    # looking at what the structure of the data is

    for X, y in train_loader:
        print(X.shape, y.shape)
        break
