"""Exercise 07: regularization."""
import torch
from torch import nn

def build_model() -> nn.Module:
    # TODO: insert dropout between layers
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(32,32),
        nn.ReLU(),
        nn.Dropout(p=0.2), 
        nn.Linear(32, 2),
    )


def build_another_model() -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Dropout(p = 0.2),
        nn.Linear(),
        nn.ReLU(),
        nn.Dropout(p = 0.2),
        nn.Linear(),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear()
    )
    
model = build_model()
# optim = torch.optim.SGD(model.parameters(), lr = 3e-4)
optim = torch.optim.Adam(model.parameters(), lr = 3e-4)
criteria = nn.CrossEntropyLoss()
# generate random data
x = torch.randn(32, 16)
y = torch.randint(0, 2, (32,))

print(x, y)

for epoch in range(100):
    model.train() # put in training mode 
    optim.zero_grad()
    logits = model(x)
    loss = criteria(logits, y)
    loss.backward()
    optim.step()

    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            output = model(x)
            print(output, epoch)
    
# TODO: when creating Adam optimizer, add weight_decay=1e-4
