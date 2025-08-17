import numpy as np
from tentorch.tensor import Tensor
from tentorch.nn.layers import Linear
from tentorch.nn.loss import MSELoss
from tentorch.nn.activations import tanh

# Define the model
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

# Create the model
model = SimpleNN()

# Create the training data
X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
y = Tensor(np.array([[0], [1], [1], [0]]))

# Define the loss function
loss_fn = MSELoss()

# Training loop
epochs = 10000
learning_rate = 0.1

for epoch in range(epochs):
    # Forward pass
    y_pred = model(X)

    # Calculate the loss
    loss = loss_fn(y_pred, y)

    # Backward pass
    loss.backward()

    # Update the weights
    for p in model.parameters():
        p.data -= learning_rate * p.grad
        p.grad = None  # Reset gradients

    if epoch % 1000 == 0:
        # The loss is a tensor with one value, so we access it with .data
        print(f"Epoch {epoch}, Loss: {loss.data}")

# Test the model
print("\n--- Test Results ---")
for i in range(4):
    test_input = X.data[i]
    expected_output = y.data[i]
    predicted_output = model(Tensor(test_input)).data
    print(f"Input: {test_input}, Expected: {expected_output}, Predicted: {predicted_output}")
