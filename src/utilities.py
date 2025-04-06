import time 
import logging
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

def add_to_class(Class):
    """
    Register functions as methods in created class
    """
    def wrapper(obj):
        setattr(Class, obj.__name__, obj) # set attribute within a class 
    return wrapper

class A:
    def __init__(self):
        self.b = 1
        
@add_to_class(A)
def do(self):
    print("Class attribute b is", self.b)

if __name__ == "__main__":
    a = A()
    a.do()

def plot_predictions(train_data, 
                     train_labels, 
                     test_data, 
                     test_labels, 
                     predictions=None):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))
  
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    
    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    
    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    
        # Show the legend
    plt.legend(prop={"size": 14});
