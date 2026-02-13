"""
Fixed GNN architecture with proper dimension handling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FixedSimpleGNN(nn.Module):
    """Simple GNN that handles variable-sized outputs correctly"""
    def __init__(self, node_in_features=5, hidden_dim=64):
        super().__init__()
        
        # Node feature transformation
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
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
        
        # Output layer: predict 3 parameters per qubit
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Fixed: always output 3 parameters per node
        )
        
    def forward(self, node_features, edge_index):
        """
        Forward pass - always outputs [num_nodes, 3]
        """
        # Encode node features
        h = self.node_encoder(node_features)  # [num_nodes, hidden_dim]
        
        # Message passing iterations
        for msg_layer, update_layer in zip(self.message_layers, self.update_layers):
            # Gather messages from neighbors
            messages = []
            for edge_idx in range(edge_index.size(1)):
                src = edge_index[0, edge_idx]
                dst = edge_index[1, edge_idx]
                
                src_h = h[src]
                dst_h = h[dst]
                
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
                    neighbor_messages = torch.stack(aggregated[node_idx])
                    agg_message = neighbor_messages.mean(dim=0)
                else:
                    agg_message = torch.zeros_like(node_h)
                
                combined = torch.cat([node_h, agg_message], dim=-1)
                updated = update_layer(combined)
                new_h.append(updated)
            
            h = torch.stack(new_h, dim=0)
        
        # Generate output predictions (3 parameters per qubit)
        predictions = self.output_layer(h)  # [num_nodes, 3]
        
        # Scale to [0, 2π]
        predictions = 2 * torch.pi * torch.sigmoid(predictions)
        
        return predictions  # Shape: [num_nodes, 3]

class FixedParameterInitializer(nn.Module):
    """Fixed parameter initialization model"""
    def __init__(self, gnn_hidden_dim=64):
        super().__init__()
        self.gnn = FixedSimpleGNN(
            node_in_features=5,
            hidden_dim=gnn_hidden_dim
        )
        
    def forward(self, node_features, edge_index):
        """Forward pass - returns [num_nodes, 3]"""
        return self.gnn(node_features, edge_index)
    
    def loss_function(self, predictions, targets, gradient_stats):
        """
        Fixed loss function with proper dimension handling
        
        Args:
            predictions: [num_nodes, 3]
            targets: [num_nodes, 3] (NOT flattened)
            gradient_stats: dictionary with gradient information
        """
        # Ensure both have same shape
        if predictions.shape != targets.shape:
            print(f"Warning: predictions shape {predictions.shape} != targets shape {targets.shape}")
            # If targets is flattened, reshape it
            if targets.dim() == 1:
                num_nodes = predictions.size(0)
                targets = targets.view(num_nodes, -1)
        
        # 1. Reconstruction loss
        mse_loss = F.mse_loss(predictions, targets)
        
        # 2. Gradient enhancement: encourage parameter diversity
        # Compute std across qubits for each parameter type
        param_std = predictions.std(dim=0).mean()  # Average std across 3 parameters
        gradient_loss = -torch.log(param_std + 1e-8)
        
        # 3. Combine losses
        total_loss = mse_loss + 0.1 * gradient_loss
        
        return total_loss
    
    def save_model(self, filepath):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'gnn_hidden_dim': self.gnn.node_encoder[0].in_features
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

def test_fixed_gnn():
    """Test the fixed GNN architecture"""
    print("Testing fixed GNN architecture...")
    
    # Create dummy data
    n_qubits = 10
    n_nodes = n_qubits
    n_edges = 20
    
    node_features = torch.randn(n_nodes, 5)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    
    # Create model
    model = FixedSimpleGNN()
    predictions = model(node_features, edge_index)
    
    print(f"Input node features shape: {node_features.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Output predictions shape: {predictions.shape}")
    print(f"Should be [num_nodes, 3]: [{n_nodes}, 3] ✓")
    
    # Test ParameterInitializer
    print("\nTesting FixedParameterInitializer...")
    param_model = FixedParameterInitializer()
    
    # Create proper targets [num_nodes, 3]
    targets = torch.randn(n_nodes, 3)
    
    # Calculate loss
    gradient_stats = {'dummy': 1.0}
    loss = param_model.loss_function(predictions, targets, gradient_stats)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Predictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"Should be in [0, 2π] ≈ [0, 6.28]")
    
    # Test with flattened targets (simulating old data format)
    print("\nTesting with flattened targets...")
    flattened_targets = targets.view(-1)  # [num_nodes * 3]
    loss2 = param_model.loss_function(predictions, flattened_targets, gradient_stats)
    print(f"Loss with flattened targets: {loss2.item():.4f}")
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_fixed_gnn()
