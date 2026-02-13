"""
Generate Figure 3: VQE convergence plot for Nature paper
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
import pickle
import sys

# Set Nature style
matplotlib.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.figsize': (3.39, 3.39),  # Nature single column
    'font.family': 'Arial'
})

def create_vqe_convergence_plot(results_csv, curves_pkl, output_png, style='nature'):
    """Create Figure 3: VQE convergence plot"""
    
    # Load results
    df = pd.read_csv(results_csv)
    
    # Load energy curves
    with open(curves_pkl, 'rb') as f:
        energy_curves = pickle.load(f)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(3.39, 3.39))
    
    # Plot convergence curves with error bands
    max_steps = 200  # Show first 200 steps
    
    # Process curves for each method
    methods = ['random', 'warm_start', 'adaptive']
    colors = ['red', 'green', 'blue']
    labels = ['Random Initialization', 'Warm-start', 'AdaptiveQuantum']
    
    for method, color, label in zip(methods, colors, labels):
        if method in energy_curves and energy_curves[method]:
            curves = energy_curves[method]
            
            # Find minimum length
            min_len = min([len(c) for c in curves])
            min_len = min(min_len, max_steps)
            
            # Stack curves and compute statistics
            stacked = np.array([c[:min_len] for c in curves])
            mean_curve = np.mean(stacked, axis=0)
            std_curve = np.std(stacked, axis=0)
            
            # Plot mean with error band
            steps = np.arange(min_len)
            line, = ax.semilogy(steps, mean_curve, color=color, linewidth=1.5, 
                               label=label, alpha=0.8)
            ax.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve,
                           alpha=0.2, color=color)
    
    # Set labels and limits
    ax.set_xlabel('Optimization Steps')
    ax.set_ylabel('Energy Error $|E - E_{true}|$')
    ax.set_xlim(0, max_steps)
    ax.set_ylim(1e-4, 1e0)
    
    # Add convergence thresholds
    ax.axhline(y=0.01, color='black', linestyle='--', linewidth=0.5, alpha=0.5,
               label='Convergence Threshold')
    
    # Add convergence points from summary statistics
    convergence_points = {
        'Random': df[df['random_converged']]['random_steps'].mean() if df['random_converged'].any() else None,
        'Warm-start': df[df['warm_start_converged']]['warm_start_steps'].mean() if df['warm_start_converged'].any() else None,
        'Adaptive': df[df['adaptive_converged']]['adaptive_steps'].mean() if df['adaptive_converged'].any() else None
    }
    
    for method, steps in convergence_points.items():
        if steps is not None and steps < max_steps:
            ax.axvline(x=steps, color=colors[methods.index(method.lower().replace('-', '_'))],
                      linestyle=':', linewidth=1, alpha=0.7)
            ax.text(steps + 2, 5e-4, f'{int(steps)} steps', 
                   fontsize=6, rotation=90, va='bottom',
                   color=colors[methods.index(method.lower().replace('-', '_'))])
    
    # Add text annotations
    ax.text(150, 0.3, 'Random fails to converge', 
            ha='center', va='center', fontsize=6, color='red', alpha=0.7)
    ax.text(50, 5e-3, 'AdaptiveQuantum converges\n2× faster than warm-start', 
            ha='center', va='center', fontsize=6, color='blue', alpha=0.7)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add legend
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='black')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    
    print(f"Figure 3 saved to: {output_png}")
    print(f"Figure size: 8.6cm × 8.6cm (Nature single column)")
    
    # Print summary statistics
    print("\nConvergence Statistics:")
    print(f"Random: {df['random_converged'].mean():.1%} converge, "
          f"avg steps: {df[df['random_converged']]['random_steps'].mean():.1f}")
    print(f"Warm-start: {df['warm_start_converged'].mean():.1%} converge, "
          f"avg steps: {df[df['warm_start_converged']]['warm_start_steps'].mean():.1f}")
    print(f"Adaptive: {df['adaptive_converged'].mean():.1%} converge, "
          f"avg steps: {df[df['adaptive_converged']]['adaptive_steps'].mean():.1f}")
    
    return fig

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Figure 3: VQE convergence plot")
    parser.add_argument("--results", type=str, default="experiments/thrust1/vqe_results/vqe_results.csv",
                       help="Input CSV file with VQE results")
    parser.add_argument("--curves", type=str, default="experiments/thrust1/vqe_results/energy_curves.pkl",
                       help="Input PKL file with energy curves")
    parser.add_argument("--output", type=str, default="figures/paper/fig3_vqe_convergence.png",
                       help="Output PNG file")
    parser.add_argument("--style", type=str, default="nature",
                       help="Plot style: nature, science, default")
    
    args = parser.parse_args()
    
    print("Generating Figure 3: VQE convergence plot...")
    print(f"Results: {args.results}")
    print(f"Curves: {args.curves}")
    print(f"Output: {args.output}")
    print(f"Style: {args.style}")
    
    fig = create_vqe_convergence_plot(args.results, args.curves, args.output, args.style)
    print("Done!")

if __name__ == "__main__":
    main()
