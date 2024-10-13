from tentorch import Tensor

a = Tensor([[1, 2], [3, 4]])
b = Tensor([[5, 6], [7, 8]])

c = Tensor.matmul(a, b)

print(c)