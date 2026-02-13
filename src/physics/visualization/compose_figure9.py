"""
Figure 9 Composite - Updated for NO PHASE TRANSITION discovery
"""

import matplotlib.pyplot as plt
from matplotlib.image import imread
import matplotlib.gridspec as gridspec
import os

plt.rcParams.update({
    'font.size': 8,
    'font.family': 'sans-serif',
    'figure.dpi': 300,
    'savefig.dpi': 600
})

def compose_figure9():
    """Create composite figure highlighting the discovery"""
    
    fig = plt.figure(figsize=(7.2, 3.6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)
    
    panel_a = imread('figures/paper/fig9_phase_diagram_3d/panel_a_phase_diagram_2d.png')
    panel_b = imread('figures/paper/fig9_phase_diagram_3d/panel_b_phase_diagram_3d.png')
    panel_c = imread('figures/paper/fig9_phase_diagram_3d/panel_c_trainability.png')
    
    ax1 = plt.subplot(gs[0])
    ax1.imshow(panel_a)
    ax1.axis('off')
    ax1.text(0.05, 0.95, 'a', transform=ax1.transAxes, fontsize=12, fontweight='bold')
    
    ax2 = plt.subplot(gs[1])
    ax2.imshow(panel_b)
    ax2.axis('off')
    ax2.text(0.05, 0.95, 'b', transform=ax2.transAxes, fontsize=12, fontweight='bold')
    
    ax3 = plt.subplot(gs[2])
    ax3.imshow(panel_c)
    ax3.axis('off')
    ax3.text(0.05, 0.95, 'c', transform=ax3.transAxes, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    os.makedirs('figures/paper/fig9_phase_diagram_3d', exist_ok=True)
    plt.savefig('figures/paper/fig9_phase_diagram_3d/fig9_composite.png', dpi=600, bbox_inches='tight')
    plt.savefig('figures/paper/fig9_phase_diagram_3d/fig9_composite.pdf', bbox_inches='tight')
    plt.close()
    
    print("✅ Figure 9 composite saved - Shows NO phase transition")

if __name__ == "__main__":
    compose_figure9()
