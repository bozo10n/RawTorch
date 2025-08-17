# RawTorch

RawTorch is a project with two primary goals:

1.  **Building a deep learning library from scratch:** The `legacy/` directory contains `TenTorch`, a deep learning library built using only Python and NumPy. This was an exercise in understanding the fundamental components of a library like PyTorch, including tensors, automatic differentiation, and neural network layers.
2.  **Exploring GPU acceleration with CUDA:** The `src/` directory contains ongoing work to extend the library's functionality with custom CUDA kernels. This is a learning exercise in high-performance computing and is currently experimental.

## Project Status

This is a personal project that I work on in my spare time. It is not intended to be a production-ready library, but rather a demonstration of my understanding of deep learning fundamentals and my journey into GPU programming.

## `legacy/` - The NumPy-based Library

The code in `legacy/src/tentorch` is a fully functional, if basic, deep learning library. It includes:

*   **`Tensor` objects:** A custom tensor implementation with automatic differentiation.
*   **`nn` module:** A collection of neural network layers (`Linear`, `ReLU`, etc.) and loss functions.
*   **`autograd`:** The automatic differentiation engine.

An example of a simple XOR network trained with this library can be found in `legacy/src/xor_demo.py`.

### Usage Example

Here is an example of how to use the `TenTorch` library to create and train a simple neural network to solve the XOR problem. This example also highlights the importance of choosing the right network architecture and activation functions.

First, we import the necessary classes:

```python
import numpy as np
from tentorch.tensor import Tensor
from tentorch.nn.layers import Linear
from tentorch.nn.loss import MSELoss
from tentorch.nn.activations import tanh
```

Next, we define our neural network. It has one hidden layer with a `tanh` activation function.

```python
class SimpleNN(object):
    def __init__(self):
        self.linear1 = Linear(2, 4)
        self.tanh = tanh()
        self.linear2 = Linear(4, 1)

    def __call__(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        return x

    def parameters(self):
        return [self.linear1.weight, self.linear1.bias, self.linear2.weight, self.linear2.bias]
```

We then create our model and the XOR dataset:

```python
model = SimpleNN()
X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
y = Tensor(np.array([[0], [1], [1], [0]]))
```

We define our loss function and the training loop. Note that the XOR problem is non-linear, so a simple network requires more epochs and a suitable learning rate to converge.

```python
loss_fn = MSELoss()
epochs = 10000
learning_rate = 0.1

for epoch in range(epochs):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    loss.backward()
    for p in model.parameters():
        p.data -= learning_rate * p.grad
        p.grad = None
```

Finally, we can see the results:

```
Epoch 0, Loss: 0.47554810841763906
Epoch 1000, Loss: 0.01655882497342341
Epoch 2000, Loss: 1.0751521693831839e-14
Epoch 3000, Loss: 3.900806242753107e-27
Epoch 4000, Loss: 2.0538116926945483e-30
Epoch 5000, Loss: 1.8119148916795115e-30
Epoch 6000, Loss: 1.1463135028992828e-30
Epoch 7000, Loss: 1.1463135028992828e-30
Epoch 8000, Loss: 1.1463135028992828e-30
Epoch 9000, Loss: 1.1463135028992828e-30

--- Test Results ---
Input: [0. 0.], Expected: [0.], Predicted: [[8.8817842e-16]] (the predicted value is so tiny that it's pretty much 0)
Input: [0. 1.], Expected: [1.], Predicted: [[1.]]
Input: [1. 0.], Expected: [1.], Predicted: [[1.]]
Input: [1. 1.], Expected: [0.], Predicted: [[1.33226763e-15]] (the predicted value is so tiny that it's pretty much 0)
```

## `src/` - CUDA Experiments

The `src/` directory contains a collection of CUDA kernels and experiments. This is where I am learning to write high-performance code for deep learning operations. The code in this section is not yet a cohesive library, but rather a collection of individual experiments.

Some of the implemented kernels include:

*   Vector addition
*   Matrix multiplication (GEMM)
*   Matrix transpose

## Future Work

The ultimate goal of this project is to integrate the CUDA kernels from `src/` into the `TenTorch` library to create a fully functional, GPU-accelerated deep learning library.

