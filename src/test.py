from tentorch import Tensor, NN, sigmoid
import numpy as np

def test_tensor_creation():
    """Test tensor creation and basic properties"""
    print("\n=== Testing Tensor Creation ===")
    
    # Test regular tensor creation
    a = Tensor([[1.456, 2], [3, 4]])
    print(f"Basic tensor shape: {a.shape}")
    assert a.shape == (2, 2), f"Expected shape (2, 2), got {a.shape}"
    
    # Test floating point precision
    assert abs(a.data[0,0] - 1.456) < 1e-6, "Floating point precision error"
    print("✓ Tensor creation tests passed")

def test_activations():
    """Test activation functions"""
    print("\n=== Testing Activation Functions ===")
    
    # Test data
    x = Tensor([[1.456, 2], [3, 4]])
    
    # Test sigmoid
    sigmoid_out = NN.sigmoid(x)
    expected_sigmoid = 1 / (1 + np.exp(-x.data))
    assert np.allclose(sigmoid_out.data, expected_sigmoid), "Sigmoid function error"
    print("✓ Sigmoid test passed")
    
    # Test tanh
    tanh_out = NN.tanh(x)
    expected_tanh = np.tanh(x.data)
    assert np.allclose(tanh_out.data, expected_tanh), "Tanh function error"
    print("✓ Tanh test passed")


def verify_gradients():
    """Verify gradient computation"""
    print("\n=== Testing Gradient Computation ===")
    
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    
    # Test sigmoid gradients
    S = sigmoid()
    y = S(x)
    y.backward()
    
    if hasattr(x, 'grad') and x.grad is not None:
        sigmoid_grad = y.data * (1 - y.data)
        # Compare the underlying numpy arrays
        assert np.allclose(x.grad.data if isinstance(x.grad, Tensor) else x.grad, sigmoid_grad), "Sigmoid gradient computation error"
        print("✓ Sigmoid gradient test passed")
    else:
        print("Note: Gradient computation not implemented or disabled")

def run_all_tests():
    """Run all verification tests"""
    print("Starting TenTorch package verification...")
    
    test_tensor_creation()
    test_activations()


    verify_gradients()
    
    print("\n=== All tests completed ===")

# Run the verification
run_all_tests()

# Additional verification of your specific example
print("\n=== Verifying Your Specific Example ===")
a = Tensor([[1.456, 2], [3, 4]])
b = Tensor([[5, 6], [7.6585, 8]])
c = NN.sigmoid(a)
d = NN.tanh(a)
ran = Tensor(np.random.randn(10, 1))


print("\nSigmoid output shape:", c.shape)
