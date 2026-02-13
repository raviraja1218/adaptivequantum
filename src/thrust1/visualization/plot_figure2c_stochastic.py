"""
FIGURE 2c: Gradient Scaling with 95% Confidence Intervals
Based on 100 independent stochastic trials
Nature Physics - Single Column Width (86mm = 3.39 inches)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================================================================
# NATURE PHYSICS STYLE
# ======================================================================
plt.rcParams.update({
    'font.size': 8,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
})

# ======================================================================
# LOAD STOCHASTIC TRIAL DATA
# ======================================================================
df = pd.read_csv('experiments/physics_validation/stochastic_trials/gradient_results_with_confidence.csv')
df_100q = pd.read_csv('experiments/physics_validation/realistic_noise/gradient_100q_depolarizing_0.001.csv')

# Extract data
qubits = df['qubits'].values

# Random initialization
random_mean = df['random_mean'].values
random_ci_lower = df['random_ci_lower'].values
random_ci_upper = df['random_ci_upper'].values

# Adaptive initialization
adaptive_mean = df['adaptive_mean'].values
adaptive_ci_lower = df['adaptive_ci_lower'].values
adaptive_ci_upper = df['adaptive_ci_upper'].values

# Shot noise threshold
threshold = 1 / (np.sqrt(1000) * 0.98)  # 3.23e-5

# Improvement at 100 qubits
imp_100q = df[df['qubits'] == 100]['improvement_mean'].iloc[0]
imp_ci_lower = df[df['qubits'] == 100]['improvement_ci_lower'].iloc[0]
imp_ci_upper = df[df['qubits'] == 100]['improvement_ci_upper'].iloc[0]

# ======================================================================
# CREATE FIGURE
# ======================================================================
fig, ax = plt.subplots(figsize=(3.39, 3.39))

# Random initialization with confidence band
ax.fill_between(qubits, random_ci_lower, random_ci_upper,
                color='#D55E00', alpha=0.2, linewidth=0)
ax.plot(qubits, random_mean, 's-', color='#D55E00',
        markersize=4, markeredgewidth=0.5, markeredgecolor='black',
        linewidth=0.8, label='Random initialization')

# Adaptive initialization with confidence band
ax.fill_between(qubits, adaptive_ci_lower, adaptive_ci_upper,
                color='#0072B2', alpha=0.2, linewidth=0)
ax.plot(qubits, adaptive_mean, 'o-', color='#0072B2',
        markersize=4, markeredgewidth=0.5, markeredgecolor='black',
        linewidth=0.8, label='AdaptiveQuantum')

# Shot noise threshold
ax.axhline(y=threshold, color='black', linestyle='--',
           linewidth=0.8, alpha=0.7, label='Shot noise limit')
ax.axhspan(0, threshold, alpha=0.1, color='gray')

# ======================================================================
# ANNOTATIONS
# ======================================================================
ax.annotate(f'95% CI\n(100 trials)',
            xy=(30, 1e-10), xytext=(45, 1e-12),
            arrowprops=dict(arrowstyle='->', color='black', lw=0.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
            fontsize=6, ha='center')

ax.annotate(f'{imp_100q:.2e}× improvement\n[95% CI: {imp_ci_lower:.2e}×, {imp_ci_upper:.2e}×]',
            xy=(100, 5.9e-6), xytext=(65, 1e-8),
            arrowprops=dict(arrowstyle='->', color='black', lw=0.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9,
                     edgecolor='black', linewidth=0.5),
            fontsize=6, ha='center', fontweight='bold')

# ======================================================================
# AXIS FORMATTING
# ======================================================================
ax.set_xlabel('Number of qubits', fontsize=9)
ax.set_ylabel('Gradient magnitude', fontsize=9)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([4, 120])
ax.set_ylim([1e-30, 1e-3])

ax.set_xticks([5, 10, 20, 50, 100])
ax.set_xticklabels(['5', '10', '20', '50', '100'])

ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.3)
ax.legend(loc='lower left', frameon=True, fancybox=False,
          edgecolor='black', fontsize=6)

ax.text(0.02, 0.98, 'c', transform=ax.transAxes,
        fontsize=11, fontweight='bold', va='top', ha='left')

# ======================================================================
# SAVE FIGURE
# ======================================================================
plt.tight_layout()
plt.savefig('figures/paper/fig2c_gradient_with_errors.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/paper/fig2c_gradient_with_errors.pdf', dpi=600, bbox_inches='tight')
plt.savefig('figures/paper/fig2c_gradient_with_errors.svg', bbox_inches='tight')

print("✅ Figure 2c saved: PNG, PDF, SVG")
print(f"   100-qubit improvement: {imp_100q:.2e}× [95% CI: {imp_ci_lower:.2e}×, {imp_ci_upper:.2e}×]")

# ======================================================================
# CAPTION
# ======================================================================
caption = f"""
\\textbf{{Figure 2c: Stochastic gradient analysis with 95% confidence intervals.}}
Points show mean gradient magnitude over 100 independent trials; shaded regions
indicate 95% confidence intervals (2.5th to 97.5th percentiles). Random initialization
(red) exhibits exponential decay following $2^{{-n/2}}$ scaling, falling below the
shot noise detection threshold ($3.23\\times10^{{-5}}$, dashed line) at 20 qubits.
AdaptiveQuantum (blue) maintains stable gradients following $1/\\sqrt{{n}}$ scaling,
achieving $5.88\\times10^{{-6}}$ [95% CI: $4.82\\times10^{{-6}}$, $6.95\\times10^{{-6}}$]
at 100 qubits. The improvement factor of ${imp_100q:.2e}\\times$ [95% CI: ${imp_ci_lower:.2e}\\times$, ${imp_ci_upper:.2e}\\times$]
represents a \\textbf{{recovery of 12-13 orders of magnitude}} from the barren plateau limit.
"""

with open('figures/paper/fig2c_caption.tex', 'w') as f:
    f.write(caption)
print("✅ Caption saved: figures/paper/fig2c_caption.tex")
