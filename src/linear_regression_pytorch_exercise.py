import torch
from torch import nn
from tqdm import tqdm

"""

----------
1. Create a straight line dataset using the linear regression formula

2. Build a pytorch model by subclassing nn.Module

3. Create a loss function and optimizer using nn.L1Loss()

4. Make predictions with the trained model on the test data
----------

Set the learning rate of the optimizer to be 0.01 and the parameters to optimize
should be the model parameters from the model you created

Write a training loop to perform the appropriate training steps for 300 epochs

The training loop should test the model on the test dataset every 20 epochs


"""


def linear_reg_example(X, weight, bias) -> None:
    """
    """
    y = weight * X + bias
    return y

class AnotherLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype = torch.float), requires_grad = True)
        self.bias = nn.Parameter(torch.randn(1, dtype = torch.float),requires_grad = True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward defines the computation in the model
        """
        return self.weights * x + self.bias

        
if __name__ == "__main__":
    weight = 0.3
    bias = 0.9 
    start = 0
    end = 1 
    step = 0.001 

    X = torch.arange(start, end, step).unsqueeze(dim = 1)
    y = linear_reg_example(X, weight, bias)
    # split the data into 80% training, 20% testing 
    train_split = int(0.8 * len(X))
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]
    

    # 2. Build a pytorch model by subclassing nn.Module
    # Set randomizer seed
    torch.manual_seed(42)
    # define the model 
    another_model_0 = AnotherLinearRegressionModel()
    # define the loss function 
    loss_fn = nn.L1Loss()
    # Create the optimizer
    optimizer = torch.optim.SGD(params = another_model_0.parameters(), lr = 0.01)
    epochs = 300 # write a training loop for 300 epochs
    # The training loop should test the model on the test dataset every 20 epochs
    train_loss_values = []
    test_loss_values = []
    epoch_count = []

    
    with torch.inference_mode():
        y_pred = another_model_0(X_train)

    for epoch in tqdm(range(epochs)):
        another_model_0.train()
        y_pred = another_model_0(X_train)

        # compute the loss
        loss = loss_fn(y_pred, y_train)
        #print(f"epoch: {epoch}, loss {loss}")
        # zero grad of the optimizer 
        optimizer.zero_grad()
        # loss backwards
        loss.backward()
        # progress the optimizer
        optimizer.step()
        # put the model in evaluation mode for testing (inference)
        another_model_0.eval()
        
        with torch.inference_mode():
            test_pred = another_model_0(X_test)
            test_loss = loss_fn(test_pred, y_test)
        if epoch % 20 == 0:
            print(f"Epoch: {epoch} train loss: {loss}, test loss: {test_loss}")
        
        
