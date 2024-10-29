import unittest
from tentorch import Tensor

class TestTensor(unittest.TestCase):

    def test_complex_computation_and_visualization(self):
        # Define tensors with requires_grad=True for autograd tracking
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True, name="a")
        b = Tensor([4.0, 5.0, 6.0], requires_grad=True, name="b")

        # Forward pass for computational chain
        c = a * b             # Element-wise multiplication
        d = c + b             # Element-wise addition
        e = d - a             # Element-wise subtraction
        f = e * e             # Element-wise square
        g = f.matmul(Tensor([[2], [2], [2]], name="matrix"))  # Matrix multiplication

        # Print out final tensor value and computation graph
        print(f"\nFinal Result Tensor g:\n{g}")
        g._visualize_graph()

        # Perform backward pass
        g.backward()

        # Assertions to confirm gradients are correctly propagated
        print("\nGradients:")
        print(f"a.grad: {a.grad}")
        print(f"b.grad: {b.grad}")

        # Verify that gradients are non-null and correctly calculated
        self.assertIsNotNone(a.grad, "Gradient for tensor 'a' should not be None.")
        self.assertIsNotNone(b.grad, "Gradient for tensor 'b' should not be None.")

        # Add more detailed checks based on manually calculated gradients if available.

if __name__ == "__main__":
    unittest.main()
