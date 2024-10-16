from tentorch import Tensor
from tentorch import AF

a = Tensor([[1.456, 2], [3, 4]])
b = Tensor([[5, 6], [7.6585, 8]])
c = AF.sigmoid(a)
d = AF.tanh(a)


print(c)
print(d)