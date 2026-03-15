import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_gnn_implementation import GCN

torch.manual_seed(12345)


dataset = TUDataset(root = 'data/TUDataset', name = 'MUTAG')
dataset = dataset.shuffle()

train_dataset = dataset[:150]
test_dataset=  dataset[150:]


train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False)

for step, data in enumerate(train_loader):
    print(data, step)
    
# Creating the model

model = GCN(hidden_channels = 64, dataset = dataset)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train() #  training mode - for example, dropout is enabled
    for data in train_loader: # Iterate in batches over the training dataset
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step() 
        optimizer.zero_grad() 
        
def test(loader):
    model.eval() # eval mode - dropout is disabled
    correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim = 1) # Use the class with the highest probability
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)


for epoch in range(1, 300):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f"Epoch: {epoch:03d}, Train acc: {train_acc:.4f}, test_acc: {test_acc:.4f}")
    
