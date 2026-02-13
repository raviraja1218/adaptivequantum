"""
Analyze Lie algebra symmetry breaking due to noise - NUMERICALLY STABLE VERSION.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_lie_algebra_symmetry():
    print("Analyzing Lie algebra symmetry breaking...")
    
    # Generate theoretical analysis - using logarithms to avoid overflow
    n_qubits_range = np.arange(5, 101, 5)
    
    results = []
    for n_qubits in n_qubits_range:
        # Theoretical dimensions - use logarithms for large numbers
        # full_dim = 4**n_qubits - 1 is too large for n>15
        log_full_dim = n_qubits * np.log(4)
        
        # Noise reduces effective dimension
        # Simple model: noise reduces accessible subspace
        noise_strength = 0.001 * n_qubits  # Increases with system size
        log_effective_dim = log_full_dim - noise_strength
        log_adaptive_dim = log_full_dim - 2 * noise_strength
        
        # Gradient scaling relation using logarithms
        # Clean system: gradients ~ 1/sqrt(full_dim)
        # Noisy system: gradients ~ 1/sqrt(effective_dim)
        log_clean_gradient = np.log(1e-5) - 0.5 * log_full_dim
        log_noisy_gradient = np.log(1e-5) - 0.5 * log_effective_dim
        log_adaptive_gradient = np.log(1e-5) - 0.5 * log_adaptive_dim
        
        # Convert back from log scale
        clean_gradient = np.exp(log_clean_gradient)
        noisy_gradient = np.exp(log_noisy_gradient)
        adaptive_gradient = np.exp(log_adaptive_gradient)
        
        improvement_factor = adaptive_gradient / clean_gradient
        symmetry_reduction = 1 - np.exp(log_adaptive_dim - log_full_dim)
        
        results.append({
            'n_qubits': n_qubits,
            'log_full_dimension': log_full_dim,
            'log_effective_dimension': log_effective_dim,
            'log_adaptive_dimension': log_adaptive_dim,
            'clean_gradient': clean_gradient,
            'noisy_gradient': noisy_gradient,
            'adaptive_gradient': adaptive_gradient,
            'improvement_factor': improvement_factor,
            'symmetry_reduction': symmetry_reduction
        })
    
    # Save results
    df = pd.DataFrame(results)
    output_dir = Path("experiments/physics_analysis/lie_algebra")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "lie_algebra_analysis_fixed.csv", index=False)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(df['n_qubits'], df['log_full_dimension'], 'b-', label='Full SU(2^n)')
    plt.plot(df['n_qubits'], df['log_effective_dimension'], 'r--', label='Noisy')
    plt.plot(df['n_qubits'], df['log_adaptive_dimension'], 'g-.', label='Adaptive')
    plt.xlabel('Number of Qubits')
    plt.ylabel('log(Lie Algebra Dimension)')
    plt.title('Symmetry Reduction by Noise (log scale)')
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
    plt.savefig(output_dir / "lie_algebra_analysis_fixed.png", dpi=300)
    plt.savefig(output_dir / "lie_algebra_analysis_fixed.pdf")
    
    print(f"✓ Lie algebra analysis saved to {output_dir}/")
    print(f"  Sample results:")
    print(f"  50 qubits: Improvement = {df[df['n_qubits']==50]['improvement_factor'].values[0]:.1e}")
    print(f"  100 qubits: Improvement = {df[df['n_qubits']==100]['improvement_factor'].values[0]:.1e}")
    
    return df

if __name__ == "__main__":
    analyze_lie_algebra_symmetry()
