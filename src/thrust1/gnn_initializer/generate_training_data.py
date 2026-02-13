"""
Generate training data for GNN-based parameter initialization
"""
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
import pickle
from pathlib import Path
from tqdm import tqdm
import sys
import argparse

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.thrust1.gnn_initializer.circuit_generator import CircuitGenerator
from src.thrust1.gnn_initializer.gradient_calculator import GradientCalculator

class GNNTrainingDataset(Dataset):
    def __init__(self, root, n_circuits=10000, max_qubits=100, min_qubits=5, 
                 transform=None, pre_transform=None):
        self.n_circuits = n_circuits
        self.max_qubits = max_qubits
        self.min_qubits = min_qubits
        self.circuit_generator = CircuitGenerator()
        self.gradient_calculator = GradientCalculator()
        
        super().__init__(root, transform, pre_transform)
        
        # Load processed data
        self.data_list = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return ['raw_data.pkl']
    
    @property
    def processed_file_names(self):
        return ['processed_data.pt']
    
    def download(self):
        # No download needed
        pass
    
    def process(self):
        print(f"Generating {self.n_circuits} training circuits...")
        data_list = []
        
        for idx in tqdm(range(self.n_circuits), desc="Generating circuits"):
            try:
                # Randomly choose number of qubits
                n_qubits = np.random.randint(self.min_qubits, self.max_qubits + 1)
                
                # Generate random circuit
                circuit_data = self.circuit_generator.generate_circuit(n_qubits)
                
                # Generate realistic noise profile
                noise_profile = self.circuit_generator.generate_noise_profile(n_qubits)
                
                # Calculate optimal initialization parameters
                optimal_params = self.gradient_calculator.find_optimal_initialization(
                    circuit_data, noise_profile
                )
                
                # Calculate gradient statistics
                gradient_stats = self.gradient_calculator.calculate_gradient_statistics(
                    circuit_data, noise_profile, optimal_params
                )
                
                # Create graph data
                graph_data = self._create_graph_data(
                    n_qubits, noise_profile, circuit_data['adjacency'], 
                    optimal_params, gradient_stats
                )
                
                data_list.append(graph_data)
                
                # Save intermediate results every 1000 circuits
                if (idx + 1) % 1000 == 0:
                    print(f"Generated {idx + 1}/{self.n_circuits} circuits")
                    
            except Exception as e:
                print(f"Error generating circuit {idx}: {e}")
                continue
        
        # Save processed data
        torch.save(data_list, self.processed_paths[0])
        print(f"Saved {len(data_list)} circuits to {self.processed_paths[0]}")
    
    def _create_graph_data(self, n_qubits, noise_profile, adjacency_matrix, 
                          optimal_params, gradient_stats):
        """Create PyTorch Geometric graph data"""
        # Node features: noise parameters for each qubit
        # [T1, T2, depolarizing_prob, dephasing_prob, gate_error_1q]
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
        
        x = torch.FloatTensor(node_features)
        
        # Edge indices from adjacency matrix
        edge_index = []
        for i in range(n_qubits):
            for j in range(n_qubits):
                if adjacency_matrix[i, j] > 0 and i != j:
                    edge_index.append([i, j])
        
        if len(edge_index) > 0:
            edge_index = torch.LongTensor(edge_index).t().contiguous()
        else:
            # If no edges, create self-loops or empty tensor
            edge_index = torch.LongTensor([[i, i] for i in range(n_qubits)]).t().contiguous()
        
        # Edge features: gate error for the connection
        edge_attr = []
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src == dst:
                # Self-loop, use average error
                error = (noise_profile['gate_error_1q'][src] + 
                        noise_profile['gate_error_1q'][dst]) / 2
            else:
                # Two-qubit connection
                error = (noise_profile['gate_error_2q'][src] + 
                        noise_profile['gate_error_2q'][dst]) / 2
            edge_attr.append([error])
        
        edge_attr = torch.FloatTensor(edge_attr)
        
        # Target: optimal initialization parameters [theta_x, theta_y, theta_z] per qubit
        y = torch.FloatTensor(optimal_params)  # Shape: [n_qubits, 3]
        
        # Additional metadata
        metadata = {
            'n_qubits': n_qubits,
            'gradient_mean': gradient_stats['mean_gradient'],
            'gradient_variance': gradient_stats['variance'],
            'trainable': gradient_stats['trainable']
        }
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                   y=y, metadata=metadata)
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]

def generate_training_data(n_circuits=10000, max_qubits=100, min_qubits=5, 
                          output_path="data/processed/gnn_training_dataset.pkl"):
    """Generate and save training dataset"""
    print(f"Generating {n_circuits} training circuits ({min_qubits}-{max_qubits} qubits)...")
    
    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    dataset = GNNTrainingDataset(
        root=str(output_dir),
        n_circuits=n_circuits,
        max_qubits=max_qubits,
        min_qubits=min_qubits
    )
    
    # Save dataset info
    dataset_info = {
        'n_circuits': len(dataset),
        'n_qubits_range': (min_qubits, max_qubits),
        'node_features': 5,  # T1, T2, depol_prob, dephasing_prob, gate_error
        'target_features': 3,  # theta_x, theta_y, theta_z per qubit
        'generation_time': pd.Timestamp.now().isoformat()
    }
    
    info_path = output_dir / "dataset_info.json"
    import json
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"Dataset generated with {len(dataset)} circuits")
    print(f"Saved to: {output_path}")
    print(f"Dataset info: {dataset_info}")
    
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GNN training data")
    parser.add_argument("--n_circuits", type=int, default=10000,
                       help="Number of training circuits to generate")
    parser.add_argument("--max_qubits", type=int, default=100,
                       help="Maximum number of qubits")
    parser.add_argument("--min_qubits", type=int, default=5,
                       help="Minimum number of qubits")
    parser.add_argument("--output", type=str, 
                       default="data/processed/gnn_training_dataset.pkl",
                       help="Output file path")
    
    args = parser.parse_args()
    
    dataset = generate_training_data(
        n_circuits=args.n_circuits,
        max_qubits=args.max_qubits,
        min_qubits=args.min_qubits,
        output_path=args.output
    )
