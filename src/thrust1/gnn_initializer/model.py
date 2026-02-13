"""
Quantum Graph Neural Network for noise-aware initialization
CORRECTED VERSION - Uses edge_index format for torch_geometric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class QuantumGNN(nn.Module):
    """
    GNN that maps hardware noise profiles → optimal initial parameters
    Input:  Noise parameters per qubit (4 features: T1, T2, depol, dephasing)
    Output: Rotation angles [θ_x, θ_y, θ_z] per qubit
    """
    
    def __init__(self, n_qubits, hidden_dim=64):
        super().__init__()
        self.n_qubits = n_qubits
        self.hidden_dim = hidden_dim
        
        # Input embedding: 4 noise params → hidden_dim
        self.embedding = nn.Linear(4, hidden_dim)
        
        # Graph convolutional layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Output layer: 3 rotation angles per qubit
        self.output = nn.Linear(hidden_dim, 3)
        
    def forward(self, noise_params, edge_index):
        """
        Args:
            noise_params: Tensor of shape (batch_size, n_qubits, 4)
            edge_index: Tensor of shape (2, num_edges) - graph connectivity
        Returns:
            angles: Tensor of shape (batch_size, n_qubits, 3)
        """
        batch_size = noise_params.shape[0]
        
        # Reshape for GNN: (batch_size * n_qubits, 4)
        x = noise_params.view(-1, 4)
        
        # Embed to hidden dimension
        x = self.embedding(x)
        x = F.relu(x)
        
        # Apply graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        
        # Output layer with tanh to bound angles in [0, 2π]
        angles = torch.tanh(self.output(x)) * torch.pi
        
        # Reshape back to (batch_size, n_qubits, 3)
        angles = angles.view(batch_size, self.n_qubits, 3)
        
        return angles

def create_linear_edge_index(n_qubits):
    """
    Create edge_index for nearest-neighbor connectivity
    Returns: torch.LongTensor of shape (2, 2*(n_qubits-1))
    """
    edges = []
    for i in range(n_qubits - 1):
        edges.append([i, i+1])
        edges.append([i+1, i])  # Bidirectional
    
    edge_index = torch.LongTensor(edges).t().contiguous()
    return edge_index

def create_mock_gnn_model(n_qubits):
    """Create a mock GNN model for testing"""
    model = QuantumGNN(n_qubits)
    return model
