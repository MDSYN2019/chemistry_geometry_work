import logging
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer as timer

# utiliites
from torch.utils.data import DataLoader

# torchvision imports
from torchvision import datasets
from torchvision.transforms import ToTensor


"""

torchvision.datasets contains a lot of example datasets you can use to practise writing computer vision conde on. FashionMNIST is one of those datasets.
Since it has 10 differet image classes, it is a multi-class classification problem

"""
BATCH_SIZE = 32


def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format).
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


# Setup training data
train_data = datasets.FashionMNIST(
    root="data",  # where to download data to?
    train=True,  # get training data
    download=True,  # download data if it doesn't exist on disk
    transform=ToTensor(),  # images come as PIL format, we want to turn into Torch tensors
    target_transform=None,  # you can transform labels as well
)

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)


# Setup testing data
test_data = datasets.FashionMNIST(
    root="data",
    train=False,  # get test data
    download=True,
    transform=ToTensor(),
)

test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
# defining the dataloaders

# Let's check out what we've created
print(f"Dataloaders: {train_dataloader, test_dataloader}")
print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")

# image, label = train_data[0]
# print(image, label)
"""
The shape of the image tensor is 1 x 28 x 28

Various problems will have variosu inputs and output shapes. But the premise remains: encode data into numbers, build a model to find patterns in those numbers,
convert those patterns into something meaningful

"""
image, label = train_data[1]
# print(f"Image shape: {image.shape}")
# plt.imshow(image.squeeze())
# plt.show()
class_names = train_data.classes


# Plot more images
# torch.manual_seed(3)
# fig = plt.figure(figsize=(9, 9))
# rows, cols = 4, 4
# for i in range(1, rows * cols + 1):
#    random_idx = torch.randint(0, len(train_data), size=[1]).item()
#    image, label = train_data[random_idx]
#    fig.add_subplot(rows, cols, i)
#    plt.imshow(image.squeeze(), cmap="gray")
#    plt.title(class_names[label])

# plt.show()
"""

The dataloader does what you think it might do - it helps load data into a model. For training and for inference,
it turns a large dataset into a python iterable of smaller chunks



These smaller chunks are called batches or mini-batches and can be set by the batch_size parameter

Why do this?

Because it's more computationally efficient.

In an ideal world you could do the forward pass and backward pass across all of your data at once

But once you start using really large datasets, unless you've got infinite computing power, it's easier to break them up into batches

It also gives your model more opportunities to improve.

With mini-batches (small portions of the data), gradient descent is performed more often per epoch (once per mini-batch rather than once per epoch).

"""

"""
Time to build a baseline model

A baseline model is one of the simplest models you can imagine.

You use the baseline as a starting point and try to improve upon it with subsequent, more complicated models.

Our baseline will consist of nn.Linear() layers.

We've done this in a previous section but there's going to be one slight difference.

"""
train_features_batch, train_labels_batch = next(iter(train_dataloader))


flatten_model = nn.Flatten()  # flattens the image into a single dimensional vector
x = train_features_batch[0]

# flatten the sample
output = flatten_model(x)
print(f"before flattening: {x.shape}")
print(f"after flattening: {output.shape}")

"""
Because we've now turned our pixel data from height and width dimensions into one long feature vector.

"""


class FashionMNISTModel(nn.Module):
    """
    input_shape = 784 - this is how many features you've got going in the model

    hidden_units = 10 - number of units/neurons in the hidden layer(s), this number could be whatever
    you want but to keep the model small we'll start with 10
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(  # I think this means just to operate on this sequentially like this
            nn.Flatten(),
            nn.Linear(in_features=input_shape, output_features=hidden_units),
            nn.Linear(in_features=input_shape, output_features=output_shape),
        )

    def forward(self, x):
        return self.layer_stack(x)


if __name__ == "__main__":
    """
    
    
    We've got all of the pirces of the puzzle readt to go, a timer, a loss ufnciton, an optimizer, a model and some data.

    Our data is now in batch form, so we'll add another loop to loop through our data batches

    Our data abtches are contained within our dataloaders, train_dataloader and test_datalaoder for the training and test data splits respectively

    A batch is BATCH_SIZE sampels of X and y, since we'ire using BATCH_SIZE = 32, out batches have 32 samples of images and targets

    Since we're computing on batches of data, our loss and evlauaiton metrics wll be calcualted per batch rather than across the whole dataset.
    
    """
    torch.manual_seed(42)
    loss_fn = nn.CrossEntropyLoss()
    # Need to setup model with input parameters
    model_0 = FashionMNISTModel(
        input_shape=784,  # one for every pixel (28x28)
        hidden_units=10,  # how many units in the hidden layer
        output_shape=len(class_names),  # one for every class
    )
    model_0.to("cpu")  # keep model on CPU to begin with

    optimizer = torch.optim.SGD(
        params=model_0.parameters(), lr=0.1
    )  # optimze the parameters, with a learning rate of 0.1
    epochs = 3
    for epoch in tqdm(range(epochs)):
        pass
