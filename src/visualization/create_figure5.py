#!/usr/bin/env python3
"""
Create Figure 5: Photon loss analysis for photonic circuits
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_photon_loss_figure():
    # Load compilation data
    data_path = Path("experiments/thrust2/final_adjusted/table2_adjusted.csv")
    df = pd.read_csv(data_path)
    
    # Calculate photon loss (0.1 photons lost per gate)
    photon_loss_rate = 0.1
    df['photons_lost'] = df['original_gates'] * photon_loss_rate
    df['photons_optimized_lost'] = df['optimized_gates'] * photon_loss_rate
    
    # Starting with 200 photons
    initial_photons = 200
    df['photons_retained'] = initial_photons - df['photons_lost']
    df['photons_optimized_retained'] = initial_photons - df['photons_optimized_lost']
    
    df['retention_rate'] = df['photons_retained'] / initial_photons
    df['optimized_retention_rate'] = df['photons_optimized_retained'] / initial_photons
    
    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Panel A: Photon retention comparison
    ax1 = axes[0]
    circuits = ['Deutsch-Jozsa', 'VQE', 'QAOA']
    x = np.arange(len(circuits))
    width = 0.35
    
    # Extract data for each circuit
    retention_original = []
    retention_optimized = []
    for circuit in circuits:
        original = df[(df['circuit'] == circuit) & (df['compiler'] == 'Qiskit')]['retention_rate'].values[0]
        optimized = df[(df['circuit'] == circuit) & (df['compiler'] == 'AdaptiveQuantum')]['optimized_retention_rate'].values[0]
        retention_original.append(original)
        retention_optimized.append(optimized)
    
    bars1 = ax1.bar(x - width/2, retention_original, width, label='Qiskit', color='#ff9999')
    bars2 = ax1.bar(x + width/2, retention_optimized, width, label='AdaptiveQuantum', color='#99ff99')
    
    ax1.set_xlabel('Circuit')
    ax1.set_ylabel('Photon Retention Rate')
    ax1.set_title('A. Photon Retention Comparison', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(circuits)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # Panel B: Gate count vs photon loss
    ax2 = axes[1]
    circuits_data = df[df['compiler'] == 'AdaptiveQuantum']
    
    x_gates = circuits_data['optimized_gates'].values
    y_photons = circuits_data['photons_optimized_retained'].values
    
    ax2.scatter(x_gates, y_photons, s=100, color='#9999ff', edgecolor='black', zorder=5)
    
    # Add labels for each circuit
    for i, (circuit, gates, photons) in enumerate(zip(circuits, x_gates, y_photons)):
        ax2.annotate(circuit, (gates, photons), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Add linear fit
    coeffs = np.polyfit(x_gates, y_photons, 1)
    x_fit = np.linspace(min(x_gates), max(x_gates), 100)
    y_fit = np.polyval(coeffs, x_fit)
    ax2.plot(x_fit, y_fit, 'r--', alpha=0.7, label=f'Fit: y={coeffs[0]:.3f}x+{coeffs[1]:.1f}')
    
    ax2.set_xlabel('Gate Count')
    ax2.set_ylabel('Photons Retained')
    ax2.set_title('B. Gate Count vs Photon Retention', fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Improvement in success probability
    ax3 = axes[2]
    
    # Calculate success probability (assuming detection efficiency)
    detection_efficiency = 0.9
    original_success = [(r * detection_efficiency) * 100 for r in retention_original]
    optimized_success = [(r * detection_efficiency) * 100 for r in retention_optimized]
    
    improvement = [(opt - orig) / orig * 100 for orig, opt in zip(original_success, optimized_success)]
    
    bars3 = ax3.bar(circuits, improvement, color=['#ffcc99', '#ccffcc', '#ccccff'])
    
    ax3.set_xlabel('Circuit')
    ax3.set_ylabel('Success Probability Improvement (%)')
    ax3.set_title('C. Success Probability Improvement', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path("figures/paper/")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / "fig5_photon_loss_analysis.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig5_photon_loss_analysis.pdf", bbox_inches='tight')
    
    print(f"✅ Figure 5 created: {output_dir / 'fig5_photon_loss_analysis.png'}")
    print(f"Photon retention data saved for Table 2")
    
    # Also save the photon loss data for Table 2
    photon_data = df[['circuit', 'compiler', 'original_gates', 'optimized_gates', 
                      'photons_retained', 'photons_optimized_retained', 'retention_rate', 'optimized_retention_rate']]
    photon_data.to_csv('experiments/thrust2/final_adjusted/photon_loss_data.csv', index=False)
    
    plt.show()
    return photon_data

if __name__ == "__main__":
    data = create_photon_loss_figure()
