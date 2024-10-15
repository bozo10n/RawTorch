from tentorch import Tensor

a = Tensor([[1.456, 2], [3, 4]])
b = Tensor([[5, 6], [7.6585, 8]])

c = Tensor.matmul(a, b)

print(c)