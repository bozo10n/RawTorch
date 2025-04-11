import unittest
import numpy as np
from tentorch import Tensor

class TestTensor(unittest.TestCase):

    def test_tensor_initialization_and_representation(self):
        tensor = Tensor([1, 2, 3])
        self.assertTrue(np.array_equal(tensor.data, np.array([1, 2, 3])))
        self.assertEqual(tensor.shape, (3,))
        self.assertEqual(repr(tensor), "Tensor([1. 2. 3.])")

    def test_addition(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([3, 4, 5], requires_grad=True)
        c = a + b
        self.assertTrue(np.array_equal(c.data, np.array([4, 6, 8])))

    def test_addition_backward(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([3, 4, 5], requires_grad=True)
        c = a + b
        c.backward()
        self.assertTrue(np.array_equal(a.grad, np.ones_like(a.data)))
        self.assertTrue(np.array_equal(b.grad, np.ones_like(b.data)))

    def test_subtraction(self):
        a = Tensor([5, 7, 9], requires_grad=True)
        b = Tensor([1, 2, 3], requires_grad=True)
        c = a - b
        self.assertTrue(np.array_equal(c.data, np.array([4, 5, 6])))

    def test_subtraction_backward(self):
        a = Tensor([5, 7, 9], requires_grad=True)
        b = Tensor([1, 2, 3], requires_grad=True)
        c = a - b
        c.backward()
        self.assertTrue(np.array_equal(a.grad, np.ones_like(a.data)))
        self.assertTrue(np.array_equal(b.grad, -np.ones_like(b.data)))

    def test_multiplication(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([4, 5, 6], requires_grad=True)
        c = a * b
        self.assertTrue(np.array_equal(c.data, np.array([4, 10, 18])))

    def test_multiplication_backward(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([4, 5, 6], requires_grad=True)
        c = a * b
        c.backward()
        self.assertTrue(np.array_equal(a.grad, b.data))
        self.assertTrue(np.array_equal(b.grad, a.data))

    def test_broadcasting(self):
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = Tensor([10, 20], requires_grad=True)
        broadcasted_b = a.broadcast(b)
        print(broadcasted_b)
        self.assertTrue(np.array_equal(broadcasted_b, np.array([[10, 20], [10, 20]])))

    def test_matmul(self):
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = Tensor([[5, 6], [7, 8]], requires_grad=True)
        c = a.matmul(b)
        self.assertTrue(np.array_equal(c.data, np.array([[19, 22], [43, 50]])))

    def test_matmul_backward(self):
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = Tensor([[5, 6], [7, 8]], requires_grad=True)
        c = a.matmul(b)
        c.backward()
        print(a.grad, np.dot(np.ones_like(c.data), b.data.T))
        print(b.grad, np.dot(a.data.T, np.ones_like(c.data)))
        self.assertTrue(np.array_equal(a.grad, np.dot(np.ones_like(c.data), b.data.T)))
        self.assertTrue(np.array_equal(b.grad, np.dot(a.data.T, np.ones_like(c.data))))

if __name__ == "__main__":
    unittest.main()
