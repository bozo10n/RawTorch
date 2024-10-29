from ..tensor import Tensor
import numpy as np
class linear:
    def __init__(self, in_features, out_features):
        self.weight = Tensor(np.randn(in_features, out_features), requires_grad=True)
        self.bias = Tensor(np.zeros((out_features,)), requires_grad = True)

    def forward(self, x):
        return x @ self.weights + self.bias    