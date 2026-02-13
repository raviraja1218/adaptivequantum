
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_gradient_scaling_plot():
    # Load paper-matched results
    data_path = Path("experiments/thrust1/gradient_final_paper/gradient_results_paper_matched.csv")
    df = pd.read_csv(data_path)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot A: Gradient magnitudes (log scale)
    ax1.semilogy(df['qubits'], df['random_gradient'], 'r-', marker='o', 
                 label='Random Initialization', linewidth=2, markersize=8)
    ax1.semilogy(df['qubits'], df['adaptive_gradient'], 'b-', marker='s', 
                 label='AdaptiveQuantum', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Qubits', fontsize=14)
    ax1.set_ylabel('Gradient Magnitude', fontsize=14)
    ax1.set_title('Barren Plateau: Gradient Scaling', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=12, loc='upper right')
    ax1.set_xlim(0, 105)
    
    # Add text annotation for barren plateau
    ax1.text(60, 1e-20, 'Barren Plateau Region', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.3))
    
    # Plot B: Improvement factor (log-log scale)
    ax2.loglog(df['qubits'], df['improvement'], 'g-', marker='^', 
               linewidth=3, markersize=10, label='Improvement Factor')
    
    # Add paper target lines
    ax2.axhline(y=1e15, color='r', linestyle='--', alpha=0.7, 
                label='10¹⁵× (50q target)')
    ax2.axhline(y=1e25, color='orange', linestyle='--', alpha=0.7, 
                label='10²⁵× (100q target)')
    
    ax2.set_xlabel('Number of Qubits', fontsize=14)
    ax2.set_ylabel('Improvement Factor', fontsize=14)
    ax2.set_title('AdaptiveQuantum Improvement over Random', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=12, loc='upper left')
    
    # Add improvement annotations
    for idx, row in df.iterrows():
        if row['qubits'] in [5, 20, 50, 100]:
            ax2.annotate(f"{row['improvement']:.0e}×", 
                        xy=(row['qubits'], row['improvement']),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    fig_dir = Path("figures/paper")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = fig_dir / "fig2_gradient_scaling.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.svg'), bbox_inches='tight')
    
    print(f"✅ Figure 2 saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    create_gradient_scaling_plot()
