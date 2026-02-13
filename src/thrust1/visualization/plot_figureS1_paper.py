"""
FIGURE S1: Non-Markovian Noise Correlations
Supplementary Information - Memory Effects in Quantum Hardware
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 8,
    'font.family': 'sans-serif',
    'figure.dpi': 300,
    'savefig.dpi': 600,
})

# Load data
df = pd.read_csv('experiments/physics_validation/non_markovian/noise_correlation_analysis.csv')

fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))

# ======================================================================
# PANEL A: Autocorrelation
# ======================================================================
ax1 = axes[0]
memory_depths = [1, 2, 5, 10]
colors = ['#999999', '#E69F00', '#56B4E9', '#009E73']

for depth, color in zip(memory_depths, colors):
    data = df[df['memory_depth'] == depth]
    ax1.plot(data['time_step'], data['autocorrelation'], 
             color=color, linewidth=1, label=f'Memory depth={depth}')
ax1.set_xlabel('Time step (Δt = 50 ns)')
ax1.set_ylabel('Autocorrelation')
ax1.set_title('a', loc='left', fontweight='bold')
ax1.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=6)
ax1.grid(True, alpha=0.2)
ax1.set_ylim([0, 1])

# ======================================================================
# PANEL B: Memory effect on gradients
# ======================================================================
ax2 = axes[1]
df_grad = pd.read_csv('experiments/physics_validation/non_markovian/gradient_decay_with_memory.csv')

for depth, color in zip(memory_depths, colors):
    data = df_grad[df_grad['memory_depth'] == depth]
    qubits_sorted = sorted(data['qubits'].unique())
    gradients = [data[data['qubits'] == q]['gradient_magnitude'].iloc[0] for q in qubits_sorted]
    ax2.semilogy(qubits_sorted, gradients, 'o-', color=color, 
                markersize=3, linewidth=0.8, label=f'Depth={depth}')
ax2.set_xlabel('Number of qubits')
ax2.set_ylabel('Gradient magnitude')
ax2.set_title('b', loc='left', fontweight='bold')
ax2.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=6)
ax2.grid(True, alpha=0.2)

# ======================================================================
# PANEL C: Markovian vs Non-Markovian
# ======================================================================
ax3 = axes[2]
df_comp = pd.read_csv('experiments/physics_validation/non_markovian/comparison_markovian_vs_nonmarkovian.csv')

ax3.semilogy(df_comp['qubits'], df_comp['markovian_gradient'], 's-', 
            color='#999999', markersize=4, label='Markovian')
ax3.semilogy(df_comp['qubits'], df_comp['non_markovian_gradient'], 'o-',
            color='#0072B2', markersize=4, label='Non-Markovian (depth=5)')
ax3.semilogy(df_comp['qubits'], df_comp['adaptive_gradient'], '^-',
            color='#D55E00', markersize=4, label='AdaptiveQuantum')
ax3.set_xlabel('Number of qubits')
ax3.set_ylabel('Gradient magnitude')
ax3.set_title('c', loc='left', fontweight='bold')
ax3.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=6)
ax3.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('figures/paper/figS1_non_markovian.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/paper/figS1_non_markovian.pdf', dpi=600, bbox_inches='tight')
print("✅ Figure S1 saved: Non-Markovian noise correlations")
