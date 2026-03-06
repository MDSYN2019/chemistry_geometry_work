import time 
import logging
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

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
