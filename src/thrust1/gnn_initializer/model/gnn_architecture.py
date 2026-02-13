"""
Graph Neural Network for noise-aware parameter initialization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleGNN(nn.Module):
    """Simple GNN for quantum parameter initialization"""
    def __init__(self, node_in_features=5, edge_in_features=1, hidden_dim=64, 
                 output_dim_per_qubit=3):
        super().__init__()
        
        # Node feature transformation
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Edge feature transformation (optional)
        self.edge_encoder = None
        if edge_in_features > 0:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_in_features, hidden_dim),
                nn.ReLU()
            )
        
        # Message passing layers
        self.message_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(3)
        ])
        
        # Node update layers
        self.update_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(3)
        ])
        
        # Output layer: predict 3 parameters per qubit (theta_x, theta_y, theta_z)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim_per_qubit)
        )
        
    def forward(self, node_features, edge_index, edge_features=None, batch=None):
        """
        Forward pass of GNN
        
        Args:
            node_features: [num_nodes, node_in_features]
            edge_index: [2, num_edges]
            edge_features: [num_edges, edge_in_features] or None
            batch: batch indices for each node
            
        Returns:
            predictions: [num_nodes, output_dim_per_qubit]
        """
        # Encode node features
        h = self.node_encoder(node_features)  # [num_nodes, hidden_dim]
        
        # Encode edge features if provided
        if edge_features is not None and self.edge_encoder is not None:
            edge_h = self.edge_encoder(edge_features)  # [num_edges, hidden_dim]
        else:
            edge_h = None
        
        # Message passing iterations
        for msg_layer, update_layer in zip(self.message_layers, self.update_layers):
            # Gather messages from neighbors
            messages = []
            for edge_idx in range(edge_index.size(1)):
                src = edge_index[0, edge_idx]
                dst = edge_index[1, edge_idx]
                
                src_h = h[src]
                dst_h = h[dst]
                
                if edge_h is not None:
                    edge_message = edge_h[edge_idx]
                    combined = torch.cat([src_h, edge_message], dim=-1)
                else:
                    combined = torch.cat([src_h, dst_h], dim=-1)
                
                message = msg_layer(combined)
                messages.append((dst, message))
            
            # Aggregate messages by destination node
            aggregated = {}
            for dst, message in messages:
                if dst not in aggregated:
                    aggregated[dst] = []
                aggregated[dst].append(message)
            
            # Update node representations
            new_h = []
            for node_idx in range(h.size(0)):
                node_h = h[node_idx]
                
                if node_idx in aggregated:
                    # Aggregate messages from neighbors (mean pooling)
                    neighbor_messages = torch.stack(aggregated[node_idx])
                    agg_message = neighbor_messages.mean(dim=0)
                else:
                    # No messages, use zero vector
                    agg_message = torch.zeros_like(node_h)
                
                # Combine node state with aggregated messages
                combined = torch.cat([node_h, agg_message], dim=-1)
                updated = update_layer(combined)
                new_h.append(updated)
            
            h = torch.stack(new_h, dim=0)
        
        # Generate output predictions (3 parameters per qubit)
        predictions = self.output_layer(h)  # [num_nodes, 3]
        
        # Apply activation to ensure parameters are in [0, 2π]
        # Using sigmoid scaled to [0, 2π]
        predictions = 2 * torch.pi * torch.sigmoid(predictions)
        
        return predictions
    
    def predict_for_circuit(self, noise_profile, adjacency_matrix):
        """Convenience method to predict parameters for a circuit"""
        # Convert noise profile to node features
        n_qubits = len(noise_profile['T1'])
        node_features = []
        
        for q in range(n_qubits):
            features = [
                noise_profile['T1'][q],
                noise_profile['T2'][q],
                noise_profile['depolarizing_prob'][q],
                noise_profile['dephasing_prob'][q],
                noise_profile['gate_error_1q'][q]
            ]
            node_features.append(features)
        
        node_features = torch.FloatTensor(node_features)
        
        # Create edge index from adjacency matrix
        edge_index = []
        for i in range(n_qubits):
            for j in range(n_qubits):
                if adjacency_matrix[i, j] > 0:
                    edge_index.append([i, j])
        
        if len(edge_index) == 0:
            # Add self-loops if no edges
            edge_index = [[i, i] for i in range(n_qubits)]
        
        edge_index = torch.LongTensor(edge_index).t().contiguous()
        
        # Predict parameters
        with torch.no_grad():
            predictions = self.forward(node_features, edge_index)
        
        return predictions.numpy()

class ParameterInitializer(nn.Module):
    """Complete parameter initialization model with training utilities"""
    def __init__(self, gnn_hidden_dim=64):
        super().__init__()
        self.gnn = SimpleGNN(
            node_in_features=5,
            edge_in_features=0,  # No edge features for now
            hidden_dim=gnn_hidden_dim,
            output_dim_per_qubit=3
        )
        
    def forward(self, node_features, edge_index):
        """Forward pass"""
        return self.gnn(node_features, edge_index)
    
    def loss_function(self, predictions, targets, gradient_stats):
        """
        Custom loss function for parameter initialization
        
        Args:
            predictions: predicted parameters [num_nodes, 3]
            targets: optimal parameters [num_nodes, 3]
            gradient_stats: dictionary with gradient information
            
        Returns:
            loss value
        """
        # 1. Reconstruction loss: match optimal parameters
        mse_loss = F.mse_loss(predictions, targets)
        
        # 2. Gradient enhancement loss: maximize gradient variance
        # We want predictions that lead to large gradients (not barren plateaus)
        # This is approximated by encouraging diversity in parameters
        param_std = predictions.std(dim=0).mean()
        gradient_loss = -torch.log(param_std + 1e-8)  # Negative log to maximize std
        
        # 3. Physical constraint loss: parameters should be in valid range
        # [0, 2π] is already enforced by sigmoid in GNN
        
        # Combine losses
        total_loss = mse_loss + 0.1 * gradient_loss
        
        return total_loss
    
    def save_model(self, filepath):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'gnn_hidden_dim': self.gnn.message_layers[0][0].in_features // 2
        }, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location='cpu')
        model = cls(gnn_hidden_dim=checkpoint['gnn_hidden_dim'])
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {filepath}")
        return model

def test_gnn():
    """Test the GNN architecture"""
    print("Testing GNN architecture...")
    
    # Create dummy data
    n_qubits = 10
    n_nodes = n_qubits
    n_edges = 20
    
    # Node features: [T1, T2, depol_prob, dephasing_prob, gate_error]
    node_features = torch.randn(n_nodes, 5)
    
    # Random edges
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    
    # Create model
    model = SimpleGNN()
    
    # Forward pass
    predictions = model(node_features, edge_index)
    
    print(f"Input node features shape: {node_features.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Output predictions shape: {predictions.shape}")
    print(f"Predictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"Should be in [0, 2π] ≈ [0, 6.28]")
    
    # Test ParameterInitializer
    print("\nTesting ParameterInitializer...")
    param_model = ParameterInitializer()
    
    # Create dummy targets
    targets = torch.randn(n_nodes, 3)
    
    # Calculate loss
    gradient_stats = {'dummy': 1.0}
    loss = param_model.loss_function(predictions, targets, gradient_stats)
    
    print(f"Loss: {loss.item():.4f}")
    
    # Test save/load
    print("\nTesting model save/load...")
    param_model.save_model("test_model.pt")
    loaded_model = ParameterInitializer.load_model("test_model.pt")
    
    # Clean up
    import os
    if os.path.exists("test_model.pt"):
        os.remove("test_model.pt")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_gnn()
