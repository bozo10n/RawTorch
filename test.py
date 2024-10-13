import torch

# Create two tensors
tensor_a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
tensor_b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

# Perform element-wise multiplication
tensor_mul = torch.matmul(tensor_a, tensor_b)

# Print the result
print("Tensor A:\n", tensor_a)
print("Tensor B:\n", tensor_b)
print("Multiplication Result (element-wise):\n", tensor_mul)
