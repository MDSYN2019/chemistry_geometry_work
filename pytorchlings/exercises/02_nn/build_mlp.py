"""Exercise 02: build a 2-layer MLP."""
import torch
from torch import nn

    
class TinyMLP(nn.Module):
    def __init__(self, in_dim: int = 4, hidden: int = 8, out_dim: int = 2):
        super().__init__()
        # TODO: replace Identity with Linear/ReLU/Linear stack
        #self.net = nn.Identity()
        self.layer_1 = nn.Linear(in_features = in_dim, out_features = hidden) # This takes the x values 'automatically'
        self.layer_2 = nn.Linear(in_features = hidden, out_features = out_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.layer_1(x)
        z = self.relu(z)
        z = self.layer_2(z)
        return z

def train_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, optim: torch.optim.Optimizer) -> float:
    # TODO: add zero_grad -> forward -> loss -> backward -> step
    pred = model(x)
    loss = nn.functional.mse_loss(pred, y)
    # optimizer zero grad
    optim.zero_grad()
    # loss backwards
    loss.backward()
    optim.step()
    return float(loss.item())


epochs = 100

model = TinyMLP()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

x = torch.randn(3,4)
y = torch.randn(3, 2)   # real target values

for epoch in range(epochs):
    loss = train_step(model, x, y, optimizer)
    print(f"We have the epoch {epoch} and loss {loss}")

    model.eval() # Changing to evaluation mode
    with torch.inference_mode():
        test_pred = model(x)
        ##print(f"test pred: {test_pred}, actual pred {y}")
        test_loss = nn.functional.mse_loss(test_pred, y)
        print(f"test loss: {test_loss.item():.4f}")
