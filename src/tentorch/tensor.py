import numpy as np

class Tensor:
    def __init__(self, data):
        self.data = np.array(data)
        self.shape = self.data.shape

    def __repr__(self):
        return f"Tensor({self.data})"

    def __add__(self, other):
        # to ensure other.data is a tensor
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)  

        return Tensor(self.data + other)      
    
    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        
        return Tensor(self.data * other)
    
    def matmul(self, other):
        if isinstance(other, Tensor):
            return Tensor(np.dot(self.data, other.data))
        
        return Tensor(np.dot(self.data, other.data))