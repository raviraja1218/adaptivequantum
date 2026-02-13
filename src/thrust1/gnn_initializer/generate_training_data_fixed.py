"""
Simplified training data generation without complex PyG dataset
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import sys
import argparse
import json
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

# Import our local modules
from src.thrust1.gnn_initializer.circuit_generator import CircuitGenerator
from src.thrust1.gnn_initializer.gradient_calculator import GradientCalculator

def generate_training_sample(n_qubits, circuit_generator, gradient_calculator):
    """Generate a single training sample"""
    try:
        # Generate random circuit
        circuit_data = circuit_generator.generate_circuit(n_qubits)
        
        # Generate noise profile (try to load from file, otherwise generate)
        noise_profile = circuit_generator.generate_noise_profile(n_qubits)
        
        # Find optimal initialization using our noise-aware method
        optimal_params = gradient_calculator.find_optimal_initialization(
            circuit_data, noise_profile, method='noise_aware'
        )
        
        # Calculate gradient statistics
        gradient_stats = gradient_calculator.calculate_gradient_statistics(
            circuit_data, noise_profile, optimal_params
        )
        
        # Create feature vector: noise parameters for each qubit
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
        
        # Create adjacency matrix edge list
        adjacency = circuit_data['adjacency']
        edge_list = []
        for i in range(n_qubits):
            for j in range(n_qubits):
                if adjacency[i, j] > 0:
                    edge_list.append([i, j])
        
        # Create sample dictionary
        sample = {
            'n_qubits': n_qubits,
            'node_features': np.array(node_features, dtype=np.float32),
            'edge_list': np.array(edge_list, dtype=np.int64),
            'optimal_params': np.array(optimal_params, dtype=np.float32),
            'gradient_stats': gradient_stats,
            'circuit_depth': circuit_data['depth'],
            'n_parameters': circuit_data['n_parameters'],
            'noise_profile': {k: np.array(v, dtype=np.float32) for k, v in noise_profile.items()}
        }
        
        return sample
    
    except Exception as e:
        print(f"Error generating sample: {e}")
        return None

def generate_dataset(n_circuits=10000, max_qubits=100, min_qubits=5, 
                    output_path="data/processed/gnn_training_dataset.pkl"):
    """Generate complete training dataset"""
    print(f"Generating {n_circuits} training circuits ({min_qubits}-{max_qubits} qubits)...")
    
    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generators
    circuit_generator = CircuitGenerator()
    gradient_calculator = GradientCalculator()
    
    # Generate samples
    samples = []
    qubit_counts = {i: 0 for i in range(min_qubits, max_qubits + 1)}
    
    with tqdm(total=n_circuits, desc="Generating circuits") as pbar:
        while len(samples) < n_circuits:
            # Randomly choose number of qubits
            n_qubits = np.random.randint(min_qubits, max_qubits + 1)
            
            # Generate sample
            sample = generate_training_sample(n_qubits, circuit_generator, gradient_calculator)
            
            if sample is not None:
                samples.append(sample)
                qubit_counts[n_qubits] += 1
                pbar.update(1)
            
            # Periodically save progress
            if len(samples) % 1000 == 0 and len(samples) > 0:
                print(f"Generated {len(samples)}/{n_circuits} samples")
    
    # Save dataset
    dataset = {
        'samples': samples,
        'metadata': {
            'n_circuits': len(samples),
            'n_qubits_range': (min_qubits, max_qubits),
            'qubit_distribution': qubit_counts,
            'node_features_dim': 5,  # T1, T2, depol_prob, dephasing_prob, gate_error
            'target_dim': samples[0]['optimal_params'].shape[0] if samples else 0,
            'generation_time': pd.Timestamp.now().isoformat(),
            'avg_gradient_norm': np.mean([s['gradient_stats']['gradient_norm'] for s in samples]),
            'trainable_fraction': np.mean([s['gradient_stats']['trainable'] for s in samples])
        }
    }
    
    # Save as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Also save metadata as JSON for easy inspection
    metadata_path = output_dir / "dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(dataset['metadata'], f, indent=2)
    
    print(f"\nDataset generation complete!")
    print(f"Saved {len(samples)} samples to {output_path}")
    print(f"Metadata saved to {metadata_path}")
    print(f"\nQubit distribution:")
    for n_q, count in sorted(qubit_counts.items()):
        if count > 0:
            print(f"  {n_q} qubits: {count} samples ({count/len(samples)*100:.1f}%)")
    
    print(f"\nAverage gradient norm: {dataset['metadata']['avg_gradient_norm']:.2e}")
    print(f"Trainable fraction: {dataset['metadata']['trainable_fraction']:.1%}")
    
    return dataset

def load_dataset(filepath):
    """Load generated dataset"""
    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GNN training data")
    parser.add_argument("--n_circuits", type=int, default=1000,  # Reduced for testing
                       help="Number of training circuits to generate")
    parser.add_argument("--max_qubits", type=int, default=100,
                       help="Maximum number of qubits")
    parser.add_argument("--min_qubits", type=int, default=5,
                       help="Minimum number of qubits")
    parser.add_argument("--output", type=str, 
                       default="data/processed/gnn_training_dataset.pkl",
                       help="Output file path")
    
    args = parser.parse_args()
    
    print("Starting GNN training data generation...")
    print(f"Configuration:")
    print(f"  Number of circuits: {args.n_circuits}")
    print(f"  Qubit range: {args.min_qubits}-{args.max_qubits}")
    print(f"  Output: {args.output}")
    print("=" * 50)
    
    dataset = generate_dataset(
        n_circuits=args.n_circuits,
        max_qubits=args.max_qubits,
        min_qubits=args.min_qubits,
        output_path=args.output
    )
