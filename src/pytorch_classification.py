
import time
import requests
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import torchvision
from torchvision import transforms, datasets
import logging

"""
The pytorch training loop

1. Forward pass - The model goes through all of the training data once, performing it's forward()
   function

2. Calculate the loss - The model's outputs are compared to the ground truth and evaluated to see how wrong
   they are

3. Zero gradients - The optimizers gradients are set to zero so they can be recalculated for the specific training step

4. Perform backpropagation on the loss - Computes the gradient of the loss with respect for every model parameter to be updated

5. 
"""

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

    
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
        """
        To make our life easier when reading from the training and test sets, we use the built-in iterator rather than
        creating one from scratch. 
        """
        train_loader = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)
        val_loader = DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle = False)
        return train_loader, val_loader

    
    
    
class BasicClassification(nn.Module):
    def __init__(self):
        """
        You need to flatten the input from the shape before feeding it to nn.linear
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer_1 = nn.Linear(in_features = 28 * 28, out_features = 128)
        self.layer_2 = nn.Linear(in_features = 128, out_features = 10)
        
    def forward(self, x):
        return self.layer_2(torch.relu(self.layer_1(self.flatten(x))))
    

if __name__ == "__main__":

    """
    Setup loss function and optimizer

    For a regression problem, you might use mean absolute error (MAE) loss
    For a binary classification problem, you'll often use binary cross entropy as the loss function
    However, the same optimizer function can often be used across different problem spaces

    ---

    We've discussed about how to take our raw outputs and convert them to prediction labels, now let's build a
    training loop

    
    """
    #device = 'cuda' if torch.cuda.is_avaliable() else 'cpu'
    epochs = 100
    data_module = FashionMNISTDataModule(batch_size = 32, resize = (28, 28))
    train_loader, val_loader = data_module.get_dataloaders()
    print(train_loader, val_loader)

    # looking at what the structure of the data is

    for X, y in train_loader:
        print(X.shape, y.shape)
        break

    model_0 = BasicClassification()
    # loss function
    #loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params = model_0.parameters(), lr= 0.1)
    train_data, train_labels = data_module.train_dataset.data, data_module.train_dataset.targets

    train_data = train_data.unsqueeze(1).float()
    y_logits = model_0(train_data)[:20]
    #print(y_logits)

    """
    This is the output of the forward() method, which implements two layers of nn.linear() which internally calls
    the following equation

    The raw outputs of this equation and in turn, the raw outputs of our model are often regarded as logits
    
    """
    y_preds_probs = torch.sigmoid(y_logits)
    #print(train_data[:20], y_preds_probs)
    
    """
    Building a training and testing loop

    Let's start training for 100 epochs and outputting the model's progress and outputting the model's progress
    every 10 epochs    
    """

    for epoch in range(epochs):
        print(f"Epoch: {epoch} ---- \n")
        train_loss = 0
        model_0.train()
        # Forward pass step 
        for X, y in train_loader:
            logits = model_0(X)
            loss = loss_fn(logits, y)
            print(loss, logits, epoch, X.shape, y.shape)
            
