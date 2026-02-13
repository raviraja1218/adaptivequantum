import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300

# Load data
df_corr = pd.read_csv('experiments/physics_validation/non_markovian/noise_correlation_analysis.csv')
df_grad = pd.read_csv('experiments/physics_validation/non_markovian/gradient_decay_with_memory.csv')
df_comp = pd.read_csv('experiments/physics_validation/non_markovian/comparison_markovian_vs_nonmarkovian.csv')

# Figure 1: Autocorrelation
fig1, ax1 = plt.subplots(figsize=(6,4))
for depth in df_corr['memory_depth'].unique():
    data = df_corr[df_corr['memory_depth'] == depth]
    ax1.plot(data['time_step'], data['autocorrelation'], label=f'Depth={depth}')
ax1.set_xlabel('Time step')
ax1.set_ylabel('Autocorrelation')
ax1.set_title('Noise Autocorrelation with Memory Effects')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/exploratory/thrust1/noise_characterization/noise_autocorrelation.png', dpi=300)
print("✅ Saved: noise_autocorrelation.png")

# Figure 2: Memory effect
fig2, ax2 = plt.subplots(figsize=(6,4))
for depth in df_grad['memory_depth'].unique():
    data = df_grad[df_grad['memory_depth'] == depth]
    ax2.plot(data['qubits'], data['gradient_magnitude'], 'o-', label=f'Memory depth={depth}')
ax2.set_xlabel('Number of qubits')
ax2.set_ylabel('Gradient magnitude')
ax2.set_yscale('log')
ax2.set_title('Effect of Memory Depth on Gradient Trainability')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/exploratory/thrust1/noise_characterization/memory_effect_on_gradients.png', dpi=300)
print("✅ Saved: memory_effect_on_gradients.png")

# Figure 3: Markovian vs Non-Markovian
fig3, ax3 = plt.subplots(figsize=(6,4))
ax3.plot(df_comp['qubits'], df_comp['markovian_gradient'], 's-', label='Markovian (memoryless)')
ax3.plot(df_comp['qubits'], df_comp['non_markovian_gradient'], 'o-', label='Non-Markovian')
ax3.set_xlabel('Number of qubits')
ax3.set_ylabel('Gradient magnitude')
ax3.set_yscale('log')
ax3.set_title('Markovian vs Non-Markovian Comparison')
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/exploratory/thrust1/noise_characterization/markovian_vs_nonmarkovian_comparison.png', dpi=300)
print("✅ Saved: markovian_vs_nonmarkovian_comparison.png")
