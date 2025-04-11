import numpy as np
import torch
import torch.nn as nn
from tentorch import Tensor
from tentorch import Linear
import matplotlib.pyplot as plt
import networkx as nx
from tentorch import MSELoss as Loss

def build_graph(tensor, G=None, visited=None, parent=None):
    if G is None:
        G = nx.DiGraph()
    if visited is None:
        visited = set()
    
    # Create unique identifier for this tensor
    tensor_id = id(tensor)
    if tensor_id not in visited:
        visited.add(tensor_id)
        # Add node with tensor shape information
        label = f"Shape: {tensor.data.shape}\nGrad: {tensor.grad is not None}"
        G.add_node(tensor_id, label=label)
        
        if parent is not None:
            G.add_edge(parent, tensor_id)
        
        # Recursively add previous tensors
        if hasattr(tensor, '_prev'):
            for prev in tensor._prev:
                build_graph(prev, G, visited, tensor_id)
    
    return G

def visualize_graph(G, title):
    plt.figure(figsize=(12, 8))
    plt.title(title)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=2000, font_size=8, font_weight='bold')
    
    # Add detailed labels
    labels = nx.get_node_attributes(G, 'label')
    pos_attrs = {}
    for node, coords in pos.items():
        pos_attrs[node] = (coords[0], coords[1] + 0.08)
    nx.draw_networkx_labels(G, pos_attrs, labels, font_size=6)
    plt.show()

def test_custom_vs_torch():
    # Hyperparameters
    batch_size = 32
    input_dim = 10
    hidden_dim = 8
    output_dim = 4
    learning_rate = 0.01

    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate random data
    X = np.random.randn(batch_size, input_dim)
    y = np.random.randn(batch_size, output_dim)

    # Custom implementation
    class CustomNet:
        def __init__(self):
            self.layer1 = Linear(input_dim, hidden_dim, seed=42)  # Add seed parameter
            self.layer2 = Linear(hidden_dim, output_dim, seed=43)  # Different seed for second layer
        
        def __call__(self, x):
            x = self.layer1(x)
            x.data = np.maximum(0, x.data)  # ReLU
            return self.layer2(x)
        
        def parameters(self):
            return [
                self.layer1.weight, self.layer1.bias,
                self.layer2.weight, self.layer2.bias
            ]

    # PyTorch implementation
    class TorchNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.layer2 = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x):
            x = torch.relu(self.layer1(x))
            return self.layer2(x)

    # Initialize networks
    custom_net = CustomNet()
    torch_net = TorchNet()

    # Get the weights from custom network
    w1, b1 = custom_net.layer1.get_weights_and_bias()
    w2, b2 = custom_net.layer2.get_weights_and_bias()

    # Initialize PyTorch network with the same weights
    with torch.no_grad():
        torch_net.layer1.weight.data = torch.tensor(w1.T, dtype=torch.float32)
        torch_net.layer1.bias.data = torch.tensor(b1, dtype=torch.float32)
        torch_net.layer2.weight.data = torch.tensor(w2.T, dtype=torch.float32)
        torch_net.layer2.bias.data = torch.tensor(b2, dtype=torch.float32)

    # Convert data to tensors
    X_tensor = Tensor(X)
    X_torch = torch.tensor(X, requires_grad=True, dtype=torch.float32)
    y_tensor = Tensor(y)
    y_torch = torch.tensor(y, dtype=torch.float32)

    # Forward pass
    custom_output = custom_net(X_tensor)
    torch_output = torch_net(X_torch)

    # Compute loss
    loss_fn = Loss()
    custom_loss = loss_fn(custom_output, y_tensor)
    torch_criterion = nn.MSELoss()
    torch_loss = torch_criterion(torch_output, y_torch)

    print("\nForward Pass Results:")
    print(f"Custom Loss: {custom_loss.data:.6f}")
    print(f"PyTorch Loss: {torch_loss.item():.6f}")
    print(f"Difference: {abs(custom_loss.data - torch_loss.item()):.6f}")

    # Print initial weights comparison
    print("\nInitial Weights Comparison:")
    print("Layer 1 weight diff:", np.abs(w1 - torch_net.layer1.weight.data.numpy().T).mean())
    print("Layer 1 bias diff:", np.abs(b1 - torch_net.layer1.bias.data.numpy()).mean())
    print("Layer 2 weight diff:", np.abs(w2 - torch_net.layer2.weight.data.numpy().T).mean())
    print("Layer 2 bias diff:", np.abs(b2 - torch_net.layer2.bias.data.numpy()).mean())

    # Backward pass
    custom_loss.backward()
    torch_loss.backward()

    # Visualize computation graphs
    custom_graph = build_graph(custom_loss)
    visualize_graph(custom_graph, "Custom Implementation Computation Graph")

    # Print gradients comparison
    print("\nGradient Comparison:")
    custom_params = custom_net.parameters()
    torch_params = list(torch_net.parameters())

    for i, (custom_p, torch_p) in enumerate(zip(custom_params, torch_params)):
        if custom_p.grad is not None and torch_p.grad is not None:
            grad_diff = np.abs(custom_p.grad - torch_p.grad.numpy().T).mean()
            print(f"Parameter {i} gradient difference: {grad_diff:.6f}")
            print(f"Custom grad norm: {np.linalg.norm(custom_p.grad):.6f}")
            print(f"Torch grad norm: {torch.norm(torch_p.grad).item():.6f}")
        else:
            print(f"Parameter {i} gradients are None")

    # Update weights
    for param in custom_params:
        if param.grad is not None:
            param.data -= learning_rate * param.grad

    with torch.no_grad():
        for param in torch_net.parameters():
            if param.grad is not None:
                param.data -= learning_rate * param.grad

    # Check updated loss
    custom_output = custom_net(X_tensor)
    torch_output = torch_net(X_torch)
    
    custom_loss_after = loss_fn(custom_output, y_tensor)
    torch_loss_after = torch_criterion(torch_output, y_torch)

    print("\nAfter Weight Update:")
    print(f"Custom Loss: {custom_loss_after.data:.6f}")
    print(f"PyTorch Loss: {torch_loss_after.item():.6f}")
    print(f"Difference: {abs(custom_loss_after.data - torch_loss_after.item()):.6f}")

    # Print final weights comparison
    print("\nFinal Weights Comparison:")
    print("Layer 1 weight diff:", np.abs(custom_net.layer1.weight.data - torch_net.layer1.weight.data.numpy().T).mean())
    print("Layer 1 bias diff:", np.abs(custom_net.layer1.bias.data - torch_net.layer1.bias.data.numpy()).mean())
    print("Layer 2 weight diff:", np.abs(custom_net.layer2.weight.data - torch_net.layer2.weight.data.numpy().T).mean())
    print("Layer 2 bias diff:", np.abs(custom_net.layer2.bias.data - torch_net.layer2.bias.data.numpy()).mean())

if __name__ == "__main__":
    test_custom_vs_torch()