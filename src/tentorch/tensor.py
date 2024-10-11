import numpy as np

class Tensor:
    def __init__(self, data):
        self.data = np.array(data)

    def __repr__(self):
        return f"{self.data}"


    def __add__(self, other):
        # to ensure other.data is a tensor
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)  

        return Tensor(self.data + other)      