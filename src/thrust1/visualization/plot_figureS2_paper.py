"""
FIGURE S2: Shot Noise Detectability Matrix
Supplementary Information - Experimental Feasibility
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8, 'figure.dpi': 300})

# Load data
df = pd.read_csv('experiments/physics_validation/shot_noise_analysis/gradient_detectability_matrix.csv')

fig, ax = plt.subplots(figsize=(4, 3))

# Prepare data for heatmap
qubits = sorted(df['qubits'].unique())
init_types = ['random', 'adaptive']
n_shots_options = [100, 500, 1000, 5000, 10000]

detectability_matrix = np.zeros((len(init_types), len(qubits)))

for i, init in enumerate(init_types):
    for j, q in enumerate(qubits):
        data = df[(df['qubits'] == q) & (df['initialization'] == init)]
        if not data.empty:
            detectability_matrix[i, j] = 1 if data['detectable_at_1000shots'].iloc[0] else 0

# Plot heatmap
im = ax.imshow(detectability_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

ax.set_xticks(np.arange(len(qubits)))
ax.set_xticklabels(qubits)
ax.set_yticks(np.arange(len(init_types)))
ax.set_yticklabels(['Random', 'Adaptive'])

ax.set_xlabel('Number of qubits')
ax.set_title('Detectability with 1000 shots, 98% fidelity\n(Green = Detectable, Red = Undetectable)')

plt.colorbar(im, ax=ax, ticks=[0, 1], label='Detectable')
plt.tight_layout()
plt.savefig('figures/paper/figS2_detectability_matrix.png', dpi=600, bbox_inches='tight')
print("✅ Figure S2 saved: Shot noise detectability matrix")
