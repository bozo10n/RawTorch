from tentorch import Tensor
from tentorch import NN
import numpy as np 

a = Tensor([[1.456, 2], [3, 4]])
b = Tensor([[5, 6], [7.6585, 8]])
c = NN.sigmoid(a)
d = NN.tanh(a)
ran = Tensor(np.random.randn(10, 1))
e = NN.linear(ran, 10, 1)


print(c)
print(e)