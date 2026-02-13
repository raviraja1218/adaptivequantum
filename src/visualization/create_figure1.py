#!/usr/bin/env python3
"""
Create Figure 1: Conceptual framework for AdaptiveQuantum
Three panels: Problem → Solution → Integrated Framework
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch, Arrow
from matplotlib.path import Path
import matplotlib.patches as patches

def create_conceptual_framework():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Panel A: Three problems
    ax1 = axes[0]
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('A. Quantum Scalability Bottlenecks', fontsize=12, fontweight='bold')
    
    # Problem 1: Barren plateaus
    ax1.add_patch(Rectangle((0.1, 0.7), 0.8, 0.2, facecolor='#ff9999', edgecolor='black'))
    ax1.text(0.5, 0.8, 'Barren Plateaus', ha='center', va='center', fontsize=10)
    ax1.text(0.5, 0.75, 'Gradient $\sim 10^{-30}$', ha='center', va='center', fontsize=8)
    
    # Problem 2: Photonic compilation
    ax1.add_patch(Rectangle((0.1, 0.4), 0.8, 0.2, facecolor='#99ff99', edgecolor='black'))
    ax1.text(0.5, 0.45, 'Photonic Compilation', ha='center', va='center', fontsize=10)
    ax1.text(0.5, 0.4, '$10^6$ gates → photon loss', ha='center', va='center', fontsize=8)
    
    # Problem 3: QEC data scarcity
    ax1.add_patch(Rectangle((0.1, 0.1), 0.8, 0.2, facecolor='#9999ff', edgecolor='black'))
    ax1.text(0.5, 0.15, 'QEC Data Scarcity', ha='center', va='center', fontsize=10)
    ax1.text(0.5, 0.1, '1000 real vs 100K needed', ha='center', va='center', fontsize=8)
    
    # Panel B: AdaptiveQuantum solutions
    ax2 = axes[1]
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('B. AdaptiveQuantum Solutions', fontsize=12, fontweight='bold')
    
    # Solution 1: GNN initialization
    ax2.add_patch(Rectangle((0.1, 0.7), 0.8, 0.2, facecolor='#ff9999', edgecolor='black'))
    ax2.text(0.5, 0.8, 'Noise-Aware GNN', ha='center', va='center', fontsize=10)
    ax2.text(0.5, 0.75, '$>10^{24}\\times$ gradient', ha='center', va='center', fontsize=8)
    
    # Solution 2: RL compiler
    ax2.add_patch(Rectangle((0.1, 0.4), 0.8, 0.2, facecolor='#99ff99', edgecolor='black'))
    ax2.text(0.5, 0.45, 'RL-Based Compiler', ha='center', va='center', fontsize=10)
    ax2.text(0.5, 0.4, '24.2% gate reduction', ha='center', va='center', fontsize=8)
    
    # Solution 3: VAE data generation
    ax2.add_patch(Rectangle((0.1, 0.1), 0.8, 0.2, facecolor='#9999ff', edgecolor='black'))
    ax2.text(0.5, 0.15, 'Physics-Informed VAE', ha='center', va='center', fontsize=10)
    ax2.text(0.5, 0.1, '70% data reduction', ha='center', va='center', fontsize=8)
    
    # Panel C: Integrated framework
    ax3 = axes[2]
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('C. Integrated Framework', fontsize=12, fontweight='bold')
    
    # Central feedback loop
    circle = plt.Circle((0.5, 0.5), 0.4, fill=False, edgecolor='black', linewidth=2)
    ax3.add_patch(circle)
    
    # Components in circle
    ax3.text(0.5, 0.75, 'Hardware Noise', ha='center', va='center', fontsize=9)
    ax3.text(0.25, 0.6, 'GNN', ha='center', va='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='#ff9999'))
    ax3.text(0.75, 0.6, 'RL Compiler', ha='center', va='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='#99ff99'))
    ax3.text(0.5, 0.35, 'VAE', ha='center', va='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='#9999ff'))
    ax3.text(0.5, 0.15, 'Quantum Circuits', ha='center', va='center', fontsize=9)
    
    # Arrows for feedback
    ax3.annotate('', xy=(0.5, 0.1), xytext=(0.5, 0.45),
                 arrowprops=dict(arrowstyle='->', color='black'))
    ax3.annotate('', xy=(0.6, 0.55), xytext=(0.85, 0.55),
                 arrowprops=dict(arrowstyle='->', color='black'))
    ax3.annotate('', xy=(0.4, 0.55), xytext=(0.15, 0.55),
                 arrowprops=dict(arrowstyle='->', color='black'))
    
    plt.tight_layout()
    plt.savefig('figures/paper/fig1_conceptual_framework.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/paper/fig1_conceptual_framework.pdf', bbox_inches='tight')
    print("✅ Figure 1 created: figures/paper/fig1_conceptual_framework.png")
    plt.show()

if __name__ == "__main__":
    create_conceptual_framework()
