"""
Generate gradient improvement data for Table 1
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import json
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.thrust1.gnn_initializer.model.gnn_architecture_fixed import FixedParameterInitializer
from src.thrust1.gnn_initializer.circuit_generator import CircuitGenerator
from src.thrust1.gnn_initializer.gradient_calculator_fixed import FixedGradientCalculator

def run_gradient_experiment_for_qubits(n_qubits, depth=20, n_trials=100, model_path=None):
    """Run gradient experiments for specific qubit count"""
    print(f"Running gradient experiments for {n_qubits} qubits...")
    
    # Initialize components
    circuit_generator = CircuitGenerator()
    gradient_calculator = FixedGradientCalculator()
    
    # Load GNN model if provided
    gnn_model = None
    if model_path and Path(model_path).exists():
        try:
            gnn_model = FixedParameterInitializer.load_model(model_path)
            gnn_model.eval()
            print(f"  Loaded GNN model from {model_path}")
        except Exception as e:
            print(f"  Failed to load GNN model: {e}")
            gnn_model = None
    
    results = []
    
    for trial in tqdm(range(n_trials), desc=f"{n_qubits} qubits", leave=False):
        try:
            # Generate circuit and noise profile
            circuit_data = circuit_generator.generate_circuit(n_qubits, depth=depth)
            noise_profile = circuit_generator.generate_noise_profile(n_qubits)
            
            # 1. Random initialization
            random_params = gradient_calculator.find_optimal_initialization(
                circuit_data, noise_profile, 'random'
            )
            random_stats = gradient_calculator.calculate_gradient_statistics(
                circuit_data, noise_profile, random_params
            )
            
            # 2. Adaptive initialization
            if gnn_model is not None:
                # Use GNN to predict parameters
                with torch.no_grad():
                    node_features = torch.FloatTensor([
                        [
                            noise_profile['T1'][q],
                            noise_profile['T2'][q],
                            noise_profile['depolarizing_prob'][q],
                            noise_profile['dephasing_prob'][q],
                            noise_profile['gate_error_1q'][q]
                        ]
                        for q in range(n_qubits)
                    ])
                    
                    # Create edge index from adjacency
                    adjacency = circuit_data['adjacency']
                    edge_list = []
                    for i in range(n_qubits):
                        for j in range(n_qubits):
                            if adjacency[i, j] > 0:
                                edge_list.append([i, j])
                    
                    if len(edge_list) == 0:
                        edge_list = [[i, i] for i in range(n_qubits)]
                    
                    edge_index = torch.LongTensor(edge_list).t().contiguous()
                    
                    # Predict parameters
                    predicted_params = gnn_model(node_features, edge_index)
                    adaptive_params = predicted_params.numpy().flatten()
            else:
                # Use noise-aware heuristic
                adaptive_params = gradient_calculator.find_optimal_initialization(
                    circuit_data, noise_profile, 'noise_aware'
                )
            
            adaptive_stats = gradient_calculator.calculate_gradient_statistics(
                circuit_data, noise_profile, adaptive_params
            )
            
            # Calculate success (trainable or not)
            success_random = random_stats['trainable']
            success_adaptive = adaptive_stats['trainable']
            
            # Calculate improvement factor
            improvement = adaptive_stats['gradient_norm'] / max(random_stats['gradient_norm'], 1e-30)
            
            results.append({
                'trial': trial,
                'n_qubits': n_qubits,
                'depth': depth,
                'random_gradient': random_stats['gradient_norm'],
                'adaptive_gradient': adaptive_stats['gradient_norm'],
                'improvement': improvement,
                'success_random': success_random,
                'success_adaptive': success_adaptive,
                'random_trainable': random_stats['trainable'],
                'adaptive_trainable': adaptive_stats['trainable']
            })
            
        except Exception as e:
            print(f"  Trial {trial} failed: {e}")
            continue
    
    if results:
        df = pd.DataFrame(results)
        
        # Calculate summary statistics
        summary = {
            'qubits': n_qubits,
            'depth': depth,
            'n_trials': len(results),
            'random_gradient_mean': df['random_gradient'].mean(),
            'random_gradient_std': df['random_gradient'].std(),
            'adaptive_gradient_mean': df['adaptive_gradient'].mean(),
            'adaptive_gradient_std': df['adaptive_gradient'].std(),
            'improvement_mean': df['improvement'].mean(),
            'improvement_std': df['improvement'].std(),
            'success_rate_random': df['success_random'].mean(),
            'success_rate_adaptive': df['success_adaptive'].mean()
        }
        
        return summary, df
    else:
        return None, None

def generate_gradient_results(qubit_sizes, depth=20, n_trials=100, model_path=None, 
                            output_dir="experiments/thrust1/gradient_data"):
    """Generate gradient results for multiple qubit sizes"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    all_summaries = []
    all_details = []
    
    for n_qubits in qubit_sizes:
        print(f"\n{'='*60}")
        print(f"Processing {n_qubits} qubits")
        print('='*60)
        
        summary, details = run_gradient_experiment_for_qubits(
            n_qubits, depth, n_trials, model_path
        )
        
        if summary is not None:
            all_summaries.append(summary)
            
            # Save detailed results for this qubit size
            if details is not None:
                details_file = Path(output_dir) / f"gradient_details_{n_qubits}q.csv"
                details.to_csv(details_file, index=False)
                all_details.append(details)
            
            print(f"\nSummary for {n_qubits} qubits:")
            print(f"  Random gradient: {summary['random_gradient_mean']:.2e} ± {summary['random_gradient_std']:.2e}")
            print(f"  Adaptive gradient: {summary['adaptive_gradient_mean']:.2e} ± {summary['adaptive_gradient_std']:.2e}")
            print(f"  Improvement: {summary['improvement_mean']:.1f}x ± {summary['improvement_std']:.1f}x")
            print(f"  Success rate (random): {summary['success_rate_random']:.1%}")
            print(f"  Success rate (adaptive): {summary['success_rate_adaptive']:.1%}")
    
    # Create consolidated summary table
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_file = Path(output_dir) / "gradient_results.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\n{'='*60}")
        print(f"Gradient experiments complete!")
        print(f"Summary saved to: {summary_file}")
        
        # Save metadata
        metadata = {
            'qubit_sizes': qubit_sizes,
            'depth': depth,
            'n_trials': n_trials,
            'model_used': model_path if model_path else 'none',
            'total_trials': len(all_summaries) * n_trials,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        metadata_file = Path(output_dir) / "experiment_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return summary_df
    else:
        print("No results generated!")
        return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run gradient improvement experiments")
    parser.add_argument("--qubits", type=str, default="5,10,15,20,25,30,40,50,75,100",
                       help="Comma-separated list of qubit sizes")
    parser.add_argument("--depth", type=int, default=20,
                       help="Circuit depth")
    parser.add_argument("--n_trials", type=int, default=100,
                       help="Number of trials per qubit size")
    parser.add_argument("--model_path", type=str, default="models/saved/gnn_initializer.pt",
                       help="Path to trained GNN model")
    parser.add_argument("--output_dir", type=str, default="experiments/thrust1/gradient_data",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Parse qubit sizes
    qubit_sizes = [int(q) for q in args.qubits.split(',')]
    
    print("Starting gradient improvement experiments...")
    print(f"Qubit sizes: {qubit_sizes}")
    print(f"Circuit depth: {args.depth}")
    print(f"Trials per size: {args.n_trials}")
    print(f"Model: {args.model_path}")
    print("=" * 60)
    
    results = generate_gradient_results(
        qubit_sizes=qubit_sizes,
        depth=args.depth,
        n_trials=args.n_trials,
        model_path=args.model_path,
        output_dir=args.output_dir
    )
    
    if results is not None:
        print("\nFinal Results Summary:")
        print(results[['qubits', 'random_gradient_mean', 'adaptive_gradient_mean', 
                      'improvement_mean', 'success_rate_random', 'success_rate_adaptive']].to_string())

if __name__ == "__main__":
    main()
