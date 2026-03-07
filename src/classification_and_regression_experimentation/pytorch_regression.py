import  time
import Requests
from pathlib import Path
import logging

import torch
from torch import nn
from tqdm import tqdm

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# 1) at the top of your script/notebook

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim = 1)
y = weight * X + bias

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

"""
Training set: The model learns from this data - 60-80% of the data

Validation set: The model gets tuned on this data: 
"""

def accuracy_fn(y_true, y_pred) -> float:
    """
    return the accuracy metric in the ML algorithm
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

class LinearRegressionModel(nn.Module):
    """
    nn.Module contains the larger building blocks (layers)

    nn.Parameter contains the smaller parameters like weights and biases

    forward() tells the larger blocks how to make calculations on inputs

    torch.optim contains optimization methods on how to improve parameters within nn.Parameter
    to better represent the data
    """
    def __init__(self):
        super().__init__() # initalize the inherited class
        self.weights = nn.Parameter(torch.randn(1, dtype = torch.float))
        self.bias = nn.Parameter(torch.randn(1, dtype = torch.float), requires_grad = True)

    def forward(self, x: torch.Tensor):
        return self.weights * x + self.bias 

class LinearRegressionModelV2(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr):
        super().__init__()
        #self.linear_layer = nn.Linear(in_features=1, out_features = 1)
        self.net = nn.Sequential(nn.LazyLinear(num_outputs))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    
def training_loop(epochs: int, model: torch.nn.Module, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer,
                  train_loss_values: list, test_loss_values: list, epoch_count: list,
                  X_train, y_train,
                  X_test, y_test, writer) -> None:
    """
    Train the model through forward propagation, computing the loss, computing the gradient, and backpropagation for
    a multiple number of times to get the true weights to get as accurate as information as possible
    """
    for epoch in tqdm(range(epochs)):
        print(f"We are on epoch {epoch}")
        model.train() # training mode
        y_pred = model(X_train)
        # Compute the loss between the predicted value and the training value
        loss = loss_fn(y_pred, y_train)
        optimizer.zero_grad()
        # Compute the loss backwards
        loss.backward()
        optimizer.step()
        
        model.eval() # evaluation mode
        with torch.inference_mode():
            # Forward pass on the test_data
            test_pred = model(X_test)

            # Compute the loss on the test data
            # Predictions come in torch.float datatype, so comparisons need to
            # be done with tensors of the same type 
            test_loss = loss_fn(test_pred, y_test.type(torch.float))                                 
            if epoch % 10 == 0:
                epoch_count.append(epoch)
                train_loss_values.append(loss.detach().numpy())
                test_loss_values.append(test_loss.detach().numpy())
                print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

        writer.add_scalar("MAE/train", loss.item(), epoch)

    writer.close()
                
    return model_0
        
        
    
if __name__ == "__main__":
    print(X, y)
    # Set manual seed since nn.Paramaeter are randomly initialized
    torch.manual_seed(30)
    model_0 = LinearRegressionModel()
    print(list(model_0.parameters()), model_0.state_dict())
   
    # Try to predict with random parameters - without any optimization or training

    #with torch.inference_mode():
    #    y_preds = model_0(X)

    """
    Creating a loss function and optimizer in PyTorch
    -------------------------------------------------

    For our model to update its parameters on its own, we'll need to add a few more things
    to our recipe

    Loss function: Measures how wrong your model's predictions are compared to the truth

    Optimizer: Tells your model how to update its internal parameters to best lower the loss


    --

    Depending on what kind of problem you have, we will have different loss functions and optimzers
    
    """
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(params = model_0.parameters(), lr = 0.01)
    
    """
    Pytorch training loop
    ---------------------

    1. Forward pass : The model goes through all of the training data once, performing it's forward()
                      function calculations

    2. Calculate the loss : The model's outputs are compared to the ground truth and evaluated to see how
                            wrong they are

    3. Zero gradients: The optimizers gradients are set to zero. So they can be recalculated for the specific training step

    4. Perform Backpropagation on the loss: Computes the gradient of the loss with respect for every model parameter.

    5. Update the Optimizer: Update the parameters with requires_grad = true

    
    
    """
    epochs = 10
    train_loss_values, test_loss_values = [], []
    train_loss_values_v2, test_loss_values_v2 = [], []
    
    epoch_count = []
    #trained_model = training_loop(100, model_0, train_loss_values, test_loss_values, epoch_count, X_train, y_train)
    #print(train_loss_values, test_loss_values, epoch_count)
    #
    #loss_fn = nn.L1Loss()
    #optimizer = torch.optim.SGD(params = model_0.parameters(), lr = 0.01)
    
    another_trained_model = LinearRegressionModelV2(1, 1, 1, lr = 0.5)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(params = another_trained_model.parameters(), lr = 0.01)    
    another_model = training_loop(100, another_trained_model, loss_fn, optimizer,
                                  train_loss_values_v2, test_loss_values_v2,
                                  epoch_count, X_train, y_train, X_test, y_test, writer = SummaryWriter(log_dir="runs/linear_experiment"))

    # Part for assessing the model 
    print("The model learned the following values for weights and bias:")
    print(another_trained_model.state_dict())
    print("\nAnd the original values for weights and bias are:")
    print(f"weights: {weight}, bias: {bias}")
