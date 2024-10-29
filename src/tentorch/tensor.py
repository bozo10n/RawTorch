import numpy as np
from numpy import dtype, float64, broadcast_to

class Tensor:
    def __init__(self, data, requires_grad=False, name=None):
        self.data = np.array(data, dtype=float64)
        self.shape = self.data.shape
        self.requires_grad= requires_grad
        self.grad = None
        self._backward = None
        self._prev = []
        self._name = name or f"Tensor_{id(self)}"

    def __repr__(self):
        return f"Tensor({self.data})"
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        queue = [self]

        while queue:
            current_tensor = queue.pop()

            if current_tensor._backward:
                current_tensor._backward()

            queue.extend(current_tensor._prev)        
    
    def broadcast(self, other):
        if isinstance(other, Tensor):
            other = other.data
        
        broadcasted_data = broadcast_to(other, self.shape)

        return broadcasted_data

    def __add__(self, other):
        # to ensure other.data is a tensor
        if isinstance(other, Tensor):
            out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        else:
            out = Tensor(self.data + other, requires_grad=self.requires_grad or other)
        

        def _backward():
            if self.grad is None:
                self.grad = np.zeros_like(self.data)

            if other.grad is None:
                other.grad = np.zeros_like(other.data)

            self.grad += np.ones_like(self.data) * out.grad
            other.grad += np.ones_like(other.data) * out.grad

        out._backward = _backward
        out._prev = [self, other]

        return out

    def __sub__(self, other):
        # to ensure other.data is a tensor
        if isinstance(other, Tensor):
            out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)
        else:
            out = Tensor(self.data - other, requires_grad=self.requires_grad or other)

        def _backward():
            if self.grad is None:
                self.grad = np.zeros_like(self.data)

            if other.grad is None:
                other.grad = np.zeros_like(other.data)

            self.grad += np.ones_like(self.data) * out.grad
            other.grad += -np.ones_like(other.data) * out.grad

        out._backward = _backward

        out._prev = [self, other]

        return out                
    
    def __mul__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        else:
            out = Tensor(self.data * other, requires_grad=self.requires_grad or other)

        def _backward():
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            if other.grad is None:
                other.grad = np.zeros_like(other.data)

            self.grad += other.data * out.grad # in respect to self
            other.grad += self.data * out.grad # in respect to out

        out._backward = _backward
        out._prev = [self, other]

        return out             
    
    def matmul(self, other):
        if isinstance(other, Tensor):
            out = Tensor(np.dot(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad)
        else:
            out = Tensor(np.dot(self.data, other), requires_grad=self.requires_grad or other)

        def _backward():
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            if other.grad is None:
                other.grad = np.zeros_like(other.data)    

            self.grad += np.dot(out.grad, other.data.T)    
            other.grad += np.dot( self.data.T, out.grad)

        out._backward = _backward
        out._prev = [self, other]

        return out    
    
    def randn(shape):
        return Tensor(np.random.randn(shape))