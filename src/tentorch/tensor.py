import numpy as np
from numpy import dtype, float64, broadcast_to

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=float64)
        self.shape = self.data.shape
        self.requires_grad= requires_grad
        self.grad = None
        self._backward = None
        self._prev = []

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
        other = self.broadcast(other)
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)  

        return Tensor(self.data + other)

    def __sub__(self, other):
        # to ensure other.data is a tensor
        other = self.broadcast(other)
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)  

        return Tensor(self.data - other)            
    
    def __mul__(self, other):
        other.data = self.broadcast(other)
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        
        return Tensor(self.data * other)
    
    def matmul(self, other):
        if isinstance(other, Tensor):
            return Tensor(np.dot(self.data, other.data))
        
        return Tensor(np.dot(self.data, other.data))
    
    def randn(shape):
        return Tensor(np.random.randn(shape))