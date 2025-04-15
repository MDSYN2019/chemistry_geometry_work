
import time
import torch
from torch import nn
from tqdm import tqdm
import torchvision
from torchvision import transforms, datasets

class FashionMNISTDataModule:
    def __init__(self, batch_size = 64, resize = (28, 28), root = './data'):
        self.batch_size = batch_size
        self.resize = resize
        self.root = root
        self._prepare_transforms()
        self._prepare_datasets()

    def _prepare_transforms(self):
        self.transform  = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor()
            ])
    
    


