import numpy as np 
from numpy import float64
from .tensor import Tensor
class sigmoid():
    def __call__(self, x):
        data = Tensor(x)
        value = 1/(1 + np.exp(-(x)))

        return value
    
class tanh():
    def __call__(self, x):
        data = Tensor(x)
        value = (np.exp(x) - np.exp(-(x))) / (np.exp(x) + np.exp(-(x)))

        return value  

class NN():
    def __init__(self, data):
        self.data = Tensor(data)

    def sigmoid(self):
        value = 1/(1 + np.exp(-(self.data)))

        return value
    
    def tanh(self):
        value = (np.exp(self.data) - np.exp(-(self.data))) / (np.exp(self.data) + np.exp(-(self.data)))

        return value
      
    def linear(input, in_features, out_features):
        weights = np.random.randn(in_features, out_features)
        biases = Tensor.randn(out_features)

        layer_output = Tensor.matmul(input, weights)
        layer_output = layer_output - biases
        return layer_output