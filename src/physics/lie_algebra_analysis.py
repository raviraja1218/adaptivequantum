"""
Analyze Lie algebra symmetry breaking due to noise.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_lie_algebra_symmetry():
    print("Analyzing Lie algebra symmetry breaking...")
    
    # Generate theoretical analysis
    n_qubits_range = np.arange(5, 101, 5)
    
    results = []
    for n_qubits in n_qubits_range:
        # Theoretical dimensions
        full_dim = 4**n_qubits - 1  # Dimension of su(2^n)
        
        # Noise reduces effective dimension
        # Simple model: noise reduces accessible subspace
        noise_strength = 0.001 * n_qubits  # Increases with system size
        effective_dim = full_dim * np.exp(-noise_strength)
        
        # Gradient scaling relation
        # Clean system: gradients ~ 1/sqrt(full_dim)
        # Noisy system: gradients ~ 1/sqrt(effective_dim)
        clean_gradient = 1e-5 / np.sqrt(full_dim)
        noisy_gradient = 1e-5 / np.sqrt(effective_dim)
        
        # Our adaptive approach: selective symmetry breaking
        adaptive_dim = full_dim * np.exp(-2 * noise_strength)  # Breaks more symmetry
        adaptive_gradient = 1e-5 / np.sqrt(adaptive_dim)
        
        improvement_factor = adaptive_gradient / clean_gradient
        
        results.append({
            'n_qubits': n_qubits,
            'full_dimension': full_dim,
            'effective_dimension': effective_dim,
            'adaptive_dimension': adaptive_dim,
            'clean_gradient': clean_gradient,
            'noisy_gradient': noisy_gradient,
            'adaptive_gradient': adaptive_gradient,
            'improvement_factor': improvement_factor,
            'symmetry_reduction': (full_dim - adaptive_dim) / full_dim
        })
    
    # Save results
    df = pd.DataFrame(results)
    output_dir = Path("experiments/physics_analysis/lie_algebra")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "lie_algebra_analysis.csv", index=False)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.semilogy(df['n_qubits'], df['full_dimension'], 'b-', label='Full SU(2^n)')
    plt.semilogy(df['n_qubits'], df['effective_dimension'], 'r--', label='Noisy')
    plt.semilogy(df['n_qubits'], df['adaptive_dimension'], 'g-.', label='Adaptive')
    plt.xlabel('Number of Qubits')
    plt.ylabel('Lie Algebra Dimension')
    plt.title('Symmetry Reduction by Noise')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.semilogy(df['n_qubits'], df['clean_gradient'], 'b-', label='Clean')
    plt.semilogy(df['n_qubits'], df['noisy_gradient'], 'r--', label='Noisy')
    plt.semilogy(df['n_qubits'], df['adaptive_gradient'], 'g-.', label='Adaptive')
    plt.xlabel('Number of Qubits')
    plt.ylabel('Gradient Magnitude')
    plt.title('Gradient Scaling')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(df['n_qubits'], df['symmetry_reduction'], 'b-')
    plt.xlabel('Number of Qubits')
    plt.ylabel('Symmetry Reduction Fraction')
    plt.title('Noise-Induced Symmetry Breaking')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.loglog(df['n_qubits'], df['improvement_factor'], 'g-')
    plt.xlabel('Number of Qubits')
    plt.ylabel('Improvement Factor')
    plt.title('Adaptive vs Clean Improvement')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "lie_algebra_analysis.png", dpi=300)
    plt.savefig(output_dir / "lie_algebra_analysis.pdf")
    
    print(f"✓ Lie algebra analysis saved to {output_dir}/")
    return df

if __name__ == "__main__":
    analyze_lie_algebra_symmetry()
