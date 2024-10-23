import numpy as np 
from numpy import float64
from .tensor import Tensor

class sigmoid:
    def __call__(self, x : Tensor):
        value = 1/(1 + np.exp(-(x.data)))

        out = Tensor(value, x.requires_grad)

        def _backward():
            # Gradient for sigmoid: s'(x) = s(x) * (1 - s(x))
            if x.grad is None:
                x.grad = Tensor(np.zeros_like(x.data))
            sigmoid_grad = out.data * (1 - out.data)
            x.grad += sigmoid_grad * out.grad

        out._backward = _backward
        out._prev = [x]    

        return out
    
class tanh():
    def __call__(self, x : Tensor):
        value = (np.exp(x) - np.exp(-(x))) / (np.exp(x) + np.exp(-(x)))
        out = Tensor(value, x.requires_grad)

        def _backward():
            
            sigmoid_grad = out.data * (1 - out.data)
            x.grad += sigmoid.grad * out.grad

        out._backward = _backward
        out._prev = [x]    

        return out

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