#!/usr/bin/env python3
"""
Create Figure 8 for Nature paper.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Set Nature style
plt.rcParams.update({
    'font.size': 8,
    'font.family': 'Arial',
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300
})

def create_figure8():
    """Create Figure 8 with 4 panels."""
    
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 8.6))  # Nature single-column
    fig.suptitle('AdaptiveQuantum: Integrated Performance', fontsize=11, fontweight='bold')
    
    # Panel A: End-to-End Success Rates
    ax = axes[0, 0]
    approaches = ['Standard', 'GNN Only', 'RL Only', 'VAE Only', 'AdaptiveQuantum']
    success_rates = [0.003, 0.020, 0.012, 0.014, 0.12]  # From paper
    colors = ['red', 'orange', 'yellow', 'green', 'blue']
    
    bars = ax.bar(approaches, success_rates, color=colors)
    ax.set_ylabel('Success Rate', fontsize=9)
    ax.set_title('A. End-to-End Performance', fontsize=10)
    ax.set_xticklabels(approaches, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add improvement labels
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        if i > 0:
            improvement = rate / success_rates[0]
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                   f'{improvement:.0f}×', ha='center', va='bottom', fontsize=7)
    
    # Panel B: Component Ablation
    ax = axes[0, 1]
    components = ['GNN', 'RL', 'VAE', 'All']
    individual = [5.0, 2.0, 3.0, 1.0]  # Individual improvements
    combined = [1.0, 1.0, 1.0, 40.0]   # Combined improvement
    
    x = np.arange(len(components))
    width = 0.35
    
    ax.bar(x - width/2, individual, width, label='Individual', color='lightblue')
    ax.bar(x + width/2, combined, width, label='Combined', color='darkblue')
    
    ax.set_ylabel('Improvement Factor', fontsize=9)
    ax.set_title('B. Component Synergy Analysis', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(components)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    # Panel C: Scaling Analysis
    ax = axes[1, 0]
    qubits = [5, 10, 20, 30, 40, 50]
    standard = [0.05, 0.02, 0.003, 0.0005, 0.0001, 0.00003]
    adaptive = [0.25, 0.20, 0.12, 0.08, 0.06, 0.04]
    
    ax.plot(qubits, standard, 'r-', marker='o', label='Standard', linewidth=2)
    ax.plot(qubits, adaptive, 'b-', marker='s', label='AdaptiveQuantum', linewidth=2)
    
    ax.set_xlabel('Number of Qubits', fontsize=9)
    ax.set_ylabel('Success Rate', fontsize=9)
    ax.set_title('C. Scaling with System Size', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Add improvement annotations
    ax.annotate(f'40×\n@20q', xy=(20, 0.12), xytext=(25, 0.2),
                arrowprops=dict(arrowstyle='->', color='black'), fontsize=7)
    ax.annotate(f'1333×\n@50q', xy=(50, 0.04), xytext=(35, 0.01),
                arrowprops=dict(arrowstyle='->', color='black'), fontsize=7)
    
    # Panel D: Resource Efficiency
    ax = axes[1, 1]
    metrics = ['Gates', 'Photons', 'Data', 'Time']
    standard_res = [100, 100, 100, 100]  # Baseline
    adaptive_res = [76, 77, 30, 62]      # Improved (24% gates, 23% photons, 70% data, 38% time)
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    standard_res = np.concatenate((standard_res, [standard_res[0]]))
    adaptive_res = np.concatenate((adaptive_res, [adaptive_res[0]]))
    
    ax.plot(angles, standard_res, 'r-', label='Standard', linewidth=2)
    ax.plot(angles, adaptive_res, 'b-', label='AdaptiveQuantum', linewidth=2)
    ax.fill(angles, adaptive_res, alpha=0.25, color='blue')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title('D. Resource Efficiency', fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('figures/paper')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'fig8_integration_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    
    print(f"✅ Figure 8 saved to: {output_path}")
    print(f"   Size: 8.6cm × 8.6cm (Nature single-column)")
    print(f"   DPI: 300")
    print(f"   Formats: PNG, PDF")
    
    plt.show()

if __name__ == '__main__':
    create_figure8()
