import numpy as np 
from numpy import float64
from ..tensor import Tensor

class sigmoid:
    def __call__(self, x : Tensor):
        value = 1/(1 + np.exp(-(x.data)))

        out = Tensor(value, x.requires_grad)

        def _backward():
            # Gradient for sigmoid: s'(x) = s(x) * (1 - s(x))
            if x.grad is None:
                x.grad = Tensor(np.zeros_like(x.data))
            sigmoid_grad = out.data * (1 - out.data)
            # chain rule
            x.grad += sigmoid_grad * out.grad

        out._backward = _backward
        out._prev = [x]    

        return out
    
class tanh():
    def __call__(self, x : Tensor):
        value = (np.exp(x.data) - np.exp(-(x.data))) / (np.exp(x.data) + np.exp(-(x.data)))
        out = Tensor(value, x.requires_grad)
        # w
        def _backward():
            if x.grad is None:
                x.grad = np.zeros_like(x.data)
            tanh_grad = 1 - (out.data * out.data)
            x.grad += tanh_grad * out.grad

        out._backward = _backward
        out._prev = [x]    

        return out