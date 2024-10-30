import numpy as np
from tentorch import Tensor

class Linear:
    def __init__(self, in_features: int, out_features: int, seed: int = 42):
        np.random.seed(seed)
        
        self.fixed_weights = np.random.randn(in_features, out_features) * 0.1
        self.fixed_bias = np.zeros(out_features)
        
        self.weight = Tensor(self.fixed_weights.copy(), requires_grad=True)
        self.bias = Tensor(self.fixed_bias.copy(), requires_grad=True)
        
        self.in_features = in_features
        self.out_features = out_features
    
    def __call__(self, x: Tensor) -> Tensor:
        weighted_sum = x.matmul(self.weight)
        
        bias_broadcasted = self.bias.data.reshape(1, -1)
        
        out = Tensor(
            weighted_sum.data + bias_broadcasted,
            requires_grad=self.weight.requires_grad or self.bias.requires_grad or x.requires_grad
        )
        
        def _backward():
            if self.weight.grad is None:
                self.weight.grad = np.zeros_like(self.weight.data)
            if self.bias.grad is None:
                self.bias.grad = np.zeros_like(self.bias.data)
            
            self.weight.grad += x.data.T @ out.grad
            
            self.bias.grad += np.sum(out.grad, axis=0)
            
            if x.requires_grad:
                if x.grad is None:
                    x.grad = np.zeros_like(x.data)
                x.grad += out.grad @ self.weight.data.T
        
        out._backward = _backward
        out._prev = [x, self.weight, self.bias]
        return out
    
    def get_weights_and_bias(self):
        return self.fixed_weights, self.fixed_bias