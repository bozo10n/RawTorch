import numpy as np 
from numpy import float64

class NN():
    def __init__(self, data):
        self.data = np.array(data, dtype=float64)

    def sigmoid(self):
        value = 1/(1 + np.exp(-(self.data)))

        return value