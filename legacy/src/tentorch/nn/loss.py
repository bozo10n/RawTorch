from ..tensor import Tensor
import numpy as np

class MSELoss():
    def __call__(self, pred : Tensor, target : Tensor):
        diff = pred - target
        out = Tensor((diff.data ** 2).mean(), requires_grad=True)

        def _backward():
            if pred.grad is None:
                pred.grad = np.zeros_like(pred.data)
            if target.grad is None:
                target.grad = np.zeros_like(target.data)

            grad = 2 * (pred.data - target.data) / target.data.size
            pred.grad += grad
            target.grad -= grad

        out._backward = _backward
        out._prev = [pred, target]

        return out     