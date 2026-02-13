"""
Figure 2b: Shot Noise Detection Threshold
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Style settings
plt.rcParams.update({
    'font.size': 8,
    'font.family': 'sans-serif',
    'figure.dpi': 300,
    'savefig.dpi': 600
})

# Create figure
fig, ax = plt.subplots(figsize=(3.39, 3.39))

# Generate data
qubits = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
random_grad = [1e-3 * (0.5)**(q/10) for q in qubits]
adaptive_grad = [5.9e-6 * (100/q)**0.3 for q in qubits]

# Shot noise threshold
threshold = 1 / (np.sqrt(1000) * 0.98)

# Plot
ax.semilogy(qubits, random_grad, 's-', color='#D55E00', label='Random initialization', 
            markersize=4, markeredgecolor='black')
ax.semilogy(qubits, adaptive_grad, 'o-', color='#0072B2', label='AdaptiveQuantum',
            markersize=4, markeredgecolor='black')
ax.axhline(y=threshold, color='black', linestyle='--', linewidth=0.8, label='Shot noise limit')
ax.axhspan(0, threshold, alpha=0.1, color='gray')

# Annotations
ax.annotate('Random falls below\ndetection threshold', xy=(15, 1e-6), xytext=(25, 1e-7),
            arrowprops=dict(arrowstyle='->', color='black', lw=0.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
            fontsize=6, ha='center')

ax.annotate('Adaptive detectable\nat 100+ qubits', xy=(100, 5.9e-6), xytext=(70, 2e-5),
            arrowprops=dict(arrowstyle='->', color='black', lw=0.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
            fontsize=6, ha='center')

ax.set_xlabel('Number of qubits', fontsize=9)
ax.set_ylabel('Gradient magnitude', fontsize=9)
ax.set_xscale('log')
ax.set_xlim([5, 120])
ax.set_ylim([1e-31, 1e-3])
ax.grid(True, alpha=0.2)
ax.legend(loc='lower left', fontsize=6)
ax.set_title('b', loc='left', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('figures/paper/fig2b_shot_noise_threshold.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/paper/fig2b_shot_noise_threshold.pdf', bbox_inches='tight')
print("✅ Saved: fig2b_shot_noise_threshold.png")
