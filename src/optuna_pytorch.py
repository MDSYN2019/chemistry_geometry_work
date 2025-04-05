import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
import optuna


# Define the dataset and dataloaders
def get_dataloaders(batch_size=64):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# Define the training and evaluation function
def train_and_evaluate(model, optimizer, epochs=3, batch_size=64, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = get_dataloaders(batch_size)

    model.to(device)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy  # This is the value that Optuna will maximize


# Define the objective function
def objective(trial):
    n_units = trial.suggest_int("n_units", 32, 512)  # Search between 32 and 512
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    model = Net(n_units)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Assume we have a training function
    accuracy = train_and_evaluate(model, optimizer)

    return accuracy  # Optuna tries to maximize this


# Define the model
class Net(nn.Module):
    def __init__(self, n_units):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, n_units)
        self.fc2 = nn.Linear(n_units, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best hyperparameters:", study.best_params)
