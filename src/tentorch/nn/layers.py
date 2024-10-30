from ..tensor import Tensor
import numpy as np

class linear:
    def __init__(self, in_features: int, out_features: int):
        self.weight = Tensor(np.random.randn(in_features, out_features) * 0.1, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)
    
    def __call__(self, x: Tensor) -> Tensor:
        # Compute weighted sum
        weighted_sum = x.matmul(self.weight)  # Shape: (batch_size, out_features)
        
        # Broadcasting the bias - reshape it to (1, out_features) for proper broadcasting
        bias_broadcasted = self.bias.data.reshape(1, -1)
        
        # Add bias with proper broadcasting
        out = Tensor(
            weighted_sum.data + bias_broadcasted,
            requires_grad=self.weight.requires_grad or self.bias.requires_grad or x.requires_grad
        )
        
        def _backward():
            if self.weight.grad is None:
                self.weight.grad = np.zeros_like(self.weight.data)
            if self.bias.grad is None:
                self.bias.grad = np.zeros_like(self.bias.data)
            
            # Gradient with respect to weights
            self.weight.grad += x.data.T @ out.grad
            
            # Gradient with respect to bias (sum across batch dimension)
            self.bias.grad += np.sum(out.grad, axis=0)
            
            # Gradient with respect to input
            if x.requires_grad:
                if x.grad is None:
                    x.grad = np.zeros_like(x.data)
                x.grad += out.grad @ self.weight.data.T
        
        out._backward = _backward
        out._prev = [x, self.weight, self.bias]
        return out