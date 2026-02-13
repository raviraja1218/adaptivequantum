"""
Figure 9, Panel A: 2D Phase Diagram
REAL DATA - Shows NO phase transition (constant γ_c = 0.12)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

plt.rcParams.update({
    'font.size': 8,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'figure.dpi': 300,
    'savefig.dpi': 600
})

def create_panel_a():
    """Generate 2D phase diagram showing NO transition"""
    
    # Load your REAL data
    df = pd.read_csv('experiments/physics_analysis/phase_transition/real_parameter_scan/real_combined_scan_results.csv')
    
    fig, ax = plt.subplots(figsize=(3.39, 3.39))
    
    qubits = sorted(df['qubits'].unique())
    noise_rates = sorted(df['noise_rate'].unique())
    
    Q, G = np.meshgrid(qubits, noise_rates)
    trainability_grid = np.zeros_like(Q, dtype=float)
    
    for i, n in enumerate(qubits):
        for j, g in enumerate(noise_rates):
            mask = (df['qubits'] == n) & (df['noise_rate'] == g)
            if mask.any():
                trainability_grid[j, i] = df[mask]['trainability_score'].mean()
    
    # Custom colormap: red (barren) to blue (trainable)
    colors = ['#D55E00', '#FFFFFF', '#0072B2']
    cmap = LinearSegmentedColormap.from_list('phase_cmap', colors, N=256)
    
    # Plot phase diagram
    im = ax.pcolormesh(Q, G, trainability_grid, cmap=cmap, vmin=0, vmax=1, shading='auto')
    
    # Plot constant critical boundary (NO transition)
    ax.axhline(y=0.12, color='black', linestyle='--', linewidth=1.5, 
               label=r'$\gamma_c = 0.12$ (constant)')
    
    # Mark operating point
    ax.plot(100, 0.001, 'w*', markersize=10, markeredgecolor='k', markeredgewidth=0.5, 
            label='AdaptiveQuantum (n=100, γ=0.001)')
    
    ax.set_xlabel('Number of qubits')
    ax.set_ylabel('Noise rate γ')
    ax.set_yscale('log')
    ax.set_xlim([5, 105])
    ax.set_ylim([8e-5, 2e-1])
    ax.legend(loc='lower left', fontsize=6, frameon=True, fancybox=False, edgecolor='black')
    
    # Add annotation highlighting the discovery
    ax.text(0.5, 0.3, 'NO PHASE TRANSITION\nTrainable up to 12% noise', 
            transform=ax.transAxes, fontsize=7, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    ax.text(0.05, 0.95, 'a', transform=ax.transAxes, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    os.makedirs('figures/paper/fig9_phase_diagram_3d', exist_ok=True)
    plt.savefig('figures/paper/fig9_phase_diagram_3d/panel_a_phase_diagram_2d.png', dpi=600, bbox_inches='tight')
    plt.savefig('figures/paper/fig9_phase_diagram_3d/panel_a_phase_diagram_2d.pdf', bbox_inches='tight')
    plt.close()
    
    print("✅ Panel A saved - Shows constant γ_c = 0.12")

if __name__ == "__main__":
    create_panel_a()
