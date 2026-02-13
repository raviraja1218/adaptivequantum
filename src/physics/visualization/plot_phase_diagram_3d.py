"""
Figure 9, Panel B: 3D Phase Diagram
REAL DATA - Shows trainability across all conditions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

plt.rcParams.update({
    'font.size': 8,
    'font.family': 'sans-serif',
    'figure.dpi': 300,
    'savefig.dpi': 600
})

def create_panel_b():
    """Generate 3D phase diagram"""
    
    df = pd.read_csv('experiments/physics_analysis/phase_transition/real_parameter_scan/real_combined_scan_results.csv')
    
    fig = plt.figure(figsize=(3.39, 3.39))
    ax = fig.add_subplot(111, projection='3d')
    
    # Prepare data
    qubits = df['qubits'].unique()
    noise_rates = df['noise_rate'].unique()
    
    X, Y = np.meshgrid(qubits, np.log10(noise_rates))
    Z = np.zeros_like(X)
    
    for i, n in enumerate(qubits):
        for j, g in enumerate(noise_rates):
            mask = (df['qubits'] == n) & (df['noise_rate'] == g)
            if mask.any():
                Z[j, i] = df[mask]['trainability_score'].mean()
    
    # Surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', vmin=0, vmax=1, alpha=0.8)
    
    # Highlight the constant trainability
    ax.text(50, -3, 0.8, f'Trainability = {Z.mean():.2f} ± {Z.std():.2f}', 
            fontsize=7, ha='center')
    
    ax.set_xlabel('Qubits')
    ax.set_ylabel('log₁₀(γ)')
    ax.set_zlabel('Trainability')
    ax.set_title('b', loc='left', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    os.makedirs('figures/paper/fig9_phase_diagram_3d', exist_ok=True)
    plt.savefig('figures/paper/fig9_phase_diagram_3d/panel_b_phase_diagram_3d.png', dpi=600, bbox_inches='tight')
    plt.savefig('figures/paper/fig9_phase_diagram_3d/panel_b_phase_diagram_3d.pdf', bbox_inches='tight')
    plt.close()
    
    print("✅ Panel B saved")

if __name__ == "__main__":
    create_panel_b()
