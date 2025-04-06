
import torch
from torch import nn
from utilities import plot_predictions

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype = torch.float))
        self.bias = nn.Parameter(torch.randn(1, dtype = torch.float), requires_grad = True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward defines the computation in the model 
        """
        return self.weights * x + self.bias

    
if __name__ == "__main__":
    # Create the loss function 
    
    # Creating some data 
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
    
    torch.manual_seed(42)
    model_0 = LinearRegressionModel()
    loss_fn = nn.L1Loss()
    # Create the optimizer
    optimizer = torch.optim.SGD(params = model_0.parameters(), lr = 0.01)
    
    print(list(model_0.parameters()))
    
    with torch.inference_mode():
        y_preds = model_0(X_test)
        
        
    # plot_predictions(train_data = X_train,  predictions = y_preds) TODO

    epochs = 100
    train_loss_values = []
    test_loss_values = []
    epoch_count = []
    for epoch in range(epochs):
        # Training
        # Put model in training mode (this is the default state of a model)
        model_0.train()
        # 1. Forward pass on train data using the forward() method inside 
        y_pred = model_0(X_train)
        # 2. Calculate the loss
        loss = loss_fn(y_pred, y_train)
        # 3. Zero grad of the optimizer
        optimizer.zero_grad()
        # 4. loss backwards 
        loss.backward()
        # 5. Progress the optimzier
        
        optimizer.step()
        # put the model in evaluation mode 
        model_0.eval()


        ### testing
        
        with torch.inference_mode():
            # Forward pass on test data
            test_pred = model_0(X_test)

            # Calculate the loss on test data
            test_loss = loss_fn(test_pred, y_test.type(torch.float))

            
            if epoch % 10 == 0:
                epoch_count.append(epoch)
                train_loss_values.append(loss.detach().numpy())
                test_loss_values.append(test_loss.detach().numpy())
                print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")
