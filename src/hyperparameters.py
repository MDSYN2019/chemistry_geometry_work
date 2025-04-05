import typing
import torch
from torch import nn
from g2l import torch as g2l
from utiliites import add_to_class


"""
We walked through various components including the data, the model, the loss function,

and the optimization algorithm.

"""


class HyperParameters:
    """
    The base class of the hyperparameters
    """
    def save_hyperparameters(self, ignore = []):
        raise NotImplemented


class B(d2l.HyperParameters):
    def __init__(self, a, b ,c):
        self.save_hyperparameters(ignore = ['c'])

class ProgressBoard(d2l.HyperParameters):
    """
    The board that plots data points in animation
    """
    def __init__(self, x_label =None, y_label = None, y_lim = None, x_scale = 'linear', y_scale = 'linear'):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n = 1):
        raise NotImplementedError


class Module(nn.Module, d2l.HyperParameters):
    """
    The base class of models

    init has the learnable parameters, the training step method accepts a data batch
    to return the loss value, and finally, the configure parameters returns the optimization method
    """
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        
    def loss(self, y_hat, y):
        raise NotImplementedError
    
    def forward(self, X) -> None:
        assert hasattr(self, 'net'), 'Neural network is defined'  # ensure we have net 
        return self.net(X)

    def plot(self, key, value, train) -> None:
        pass

    def training_step(self, batch) -> None:
        l = self.loss(self(*batch[:-1]), batch[-1])
        return l

    def validation_step(self, batch):
        pass

    def configure_optimizers(self):
        pass
    


class DataModule(d2l.HyperParameters):
    """
    The DataModule class is the base class for data. Quite frequently the __init__ method
    is used to prepare the data. This includes downloading and preprocessing if needed. The train
    dataloader returns the dataloader for the training ataset. A data loader is a (python)
    generator that yields a data batch each time it is used.

    This batch is then fed into the training the training method of the module to compute the loss.

    There is an optional val_dataloader to return the validation dataset loader.
        
    """


    def __init__(self, root = '../data', num_workers = 1):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train = True)
    
    def val_dataloader(self):
        return self.get_dataloader(train = True)


class Trainer(d2l.HyperParameters):
    """

    The Trainer class trains the learnable parameters in the Module
    class with data specified in DataModule. The key method is fit,
    which accepts two arguments: model, an instance of Module, and
    data, an instance of DataModule. It then iterates over the entire
    dataset max_epochs times to train the model. As before, we will
    defer the implementation of this method to later chapters.
    
    """
    def __init__(self, max_epochs, num_gpus = 0, gradient_clip_val = 0):
        self.save_hyperparameters()
        assert num_gpus == 0,
    

# This is a bad implementation of the above class, which needs an improvement

class SimpleDataModule(DataModule):
    """
    Get a simple data loader module here
    """
    def __init___(self):
        super().__init__() # initalize all the methods within DataModule, therefore the DataModule
    def get_dataloader(self, train):
        batch_size = 0
        train_dataset_size = len(train)
        for i in range(0, train_dataset_size, batch_size):
            yield train[i, i+batch_size]
    



                
