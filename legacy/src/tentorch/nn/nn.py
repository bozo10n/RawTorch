import numpy as np 
from numpy import float64
from ..tensor import Tensor


class NN():
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x