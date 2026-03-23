import torch

logits = torch.tensor([
    [2.1, 0.3],
    [0.2, 1.7],
    [1.0, 0.9],
])

y = torch.tensor([0,1,1])

pred = logits.argmax(dim = 1)
correct = (pred == y)
acc = correct.float().mean()
print(acc.item)
