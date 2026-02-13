"""
FINAL fixed training data generation with proper gradient scaling
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
from src.thrust1.gnn_initializer.gradient_calculator_final import FinalGradientCalculator

def generate_training_sample(n_qubits, circuit_generator, gradient_calculator):
    """Generate a single training sample with proper gradient scaling"""
    try:
        # Generate random circuit
        circuit_data = circuit_generator.generate_circuit(n_qubits, depth=10)
        
        # Generate noise profile
        noise_profile = circuit_generator.generate_noise_profile(n_qubits)
        
        # Find optimal initialization using noise-aware method
        optimal_params = gradient_calculator.find_optimal_initialization(
            circuit_data, noise_profile, method='noise_aware'
        )
        
        # Calculate gradient statistics for ADAPTIVE method
        adaptive_stats = gradient_calculator.calculate_gradient_statistics(
            circuit_data, noise_profile, optimal_params, method='adaptive'
        )
        
        # Also calculate for RANDOM method for comparison
        random_params = gradient_calculator.find_optimal_initialization(
            circuit_data, noise_profile, method='random'
        )
        random_stats = gradient_calculator.calculate_gradient_statistics(
            circuit_data, noise_profile, random_params, method='random'
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
        
        # Ensure we have edges (add self-loops if none)
        if len(edge_list) == 0:
            edge_list = [[i, i] for i in range(n_qubits)]
        
        # Create sample dictionary
        sample = {
            'n_qubits': n_qubits,
            'node_features': np.array(node_features, dtype=np.float32),
            'edge_list': np.array(edge_list, dtype=np.int64),
            'optimal_params': np.array(optimal_params, dtype=np.float32),
            'gradient_stats': {
                'adaptive': adaptive_stats,
                'random': random_stats
            },
            'circuit_depth': circuit_data['depth'],
            'improvement_factor': adaptive_stats['improvement_factor'] / max(random_stats['improvement_factor'], 1)
        }
        
        return sample
    
    except Exception as e:
        print(f"Error generating sample: {e}")
        return None

def generate_final_dataset_fixed(n_circuits=10000, max_qubits=100, min_qubits=5, 
                                output_path="data/processed/gnn_training_dataset_final_fixed.pkl"):
    """Generate complete training dataset with proper gradient scaling"""
    print(f"Generating {n_circuits} training circuits ({min_qubits}-{max_qubits} qubits)...")
    
    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generators
    circuit_generator = CircuitGenerator()
    gradient_calculator = FinalGradientCalculator()
    
    # Generate samples
    samples = []
    qubit_counts = {i: 0 for i in range(min_qubits, max_qubits + 1)}
    improvement_factors = []
    
    with tqdm(total=n_circuits, desc="Generating circuits") as pbar:
        while len(samples) < n_circuits:
            # Randomly choose number of qubits
            n_qubits = np.random.randint(min_qubits, max_qubits + 1)
            
            # Generate sample
            sample = generate_training_sample(n_qubits, circuit_generator, gradient_calculator)
            
            if sample is not None:
                # Verify shape
                expected_params = n_qubits * 3
                actual_params = len(sample['optimal_params'])
                
                if actual_params == expected_params:
                    samples.append(sample)
                    qubit_counts[n_qubits] += 1
                    improvement_factors.append(sample['improvement_factor'])
                    pbar.update(1)
                else:
                    print(f"Warning: Expected {expected_params} params, got {actual_params}")
            
            # Periodically save progress
            if len(samples) % 1000 == 0 and len(samples) > 0:
                print(f"Generated {len(samples)}/{n_circuits} samples")
                print(f"Average improvement factor: {np.mean(improvement_factors):.1e}x")
    
    # Save dataset
    dataset = {
        'samples': samples,
        'metadata': {
            'n_circuits': len(samples),
            'n_qubits_range': (min_qubits, max_qubits),
            'qubit_distribution': qubit_counts,
            'node_features_dim': 5,
            'target_dim_per_qubit': 3,
            'generation_time': pd.Timestamp.now().isoformat(),
            'avg_improvement_factor': float(np.mean(improvement_factors)),
            'max_improvement_factor': float(np.max(improvement_factors)),
            'min_improvement_factor': float(np.min(improvement_factors)),
            'avg_adaptive_gradient': np.mean([s['gradient_stats']['adaptive']['gradient_norm'] for s in samples]),
            'avg_random_gradient': np.mean([s['gradient_stats']['random']['gradient_norm'] for s in samples]),
            'adaptive_trainable_fraction': np.mean([s['gradient_stats']['adaptive']['trainable'] for s in samples]),
            'random_trainable_fraction': np.mean([s['gradient_stats']['random']['trainable'] for s in samples])
        }
    }
    
    # Save as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Also save metadata as JSON for easy inspection
    metadata_path = output_dir / "dataset_metadata_final_fixed.json"
    with open(metadata_path, 'w') as f:
        json.dump(dataset['metadata'], f, indent=2)
    
    print(f"\nDataset generation complete!")
    print(f"Saved {len(samples)} samples to {output_path}")
    print(f"Metadata saved to {metadata_path}")
    
    # Print summary
    print(f"\nKey Statistics:")
    print(f"  Average improvement factor: {dataset['metadata']['avg_improvement_factor']:.1e}x")
    print(f"  Max improvement factor: {dataset['metadata']['max_improvement_factor']:.1e}x")
    print(f"  Adaptive trainable fraction: {dataset['metadata']['adaptive_trainable_fraction']:.1%}")
    print(f"  Random trainable fraction: {dataset['metadata']['random_trainable_fraction']:.1%}")
    
    print(f"\nQubit distribution (top 10):")
    sorted_counts = sorted(qubit_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for n_q, count in sorted_counts:
        if count > 0:
            print(f"  {n_q} qubits: {count} samples ({count/len(samples)*100:.1f}%)")
    
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Generate FINAL training data with proper gradient scaling")
    parser.add_argument("--n_circuits", type=int, default=2000,
                       help="Number of training circuits to generate")
    parser.add_argument("--max_qubits", type=int, default=100,
                       help="Maximum number of qubits")
    parser.add_argument("--min_qubits", type=int, default=5,
                       help="Minimum number of qubits")
    parser.add_argument("--output", type=str, 
                       default="data/processed/gnn_training_dataset_fixed.pkl",
                       help="Output file path")
    
    args = parser.parse_args()
    
    print("Starting FINAL GNN training data generation...")
    print(f"Configuration:")
    print(f"  Number of circuits: {args.n_circuits}")
    print(f"  Qubit range: {args.min_qubits}-{args.max_qubits}")
    print(f"  Output: {args.output}")
    print("=" * 50)
    
    dataset = generate_final_dataset_fixed(
        n_circuits=args.n_circuits,
        max_qubits=args.max_qubits,
        min_qubits=args.min_qubits,
        output_path=args.output
    )

if __name__ == "__main__":
    main()
