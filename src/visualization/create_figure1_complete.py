"""
Create Figure 1: Conceptual framework for Nature Physics - COMPLETE VERSION.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

def create_conceptual_framework():
    print("Creating Figure 1: Conceptual framework...")
    
    # Set Nature-style parameters
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'font.size': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.figsize': (8.6/2.54, 8.6/2.54),  # 8.6cm in inches
    })
    
    fig, ax = plt.subplots(figsize=(8.6/2.54, 8.6/2.54))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors for Nature Physics
    colors = {
        'problem': '#d62728',
        'solution': '#2ca02c',
        'result': '#1f77b4',
        'arrow': '#7f7f7f'
    }
    
    # Problem boxes
    problem_y = [7, 4.5, 2]
    problem_texts = [
        'Barren Plateaus:\nGradients ∼ 2⁻ⁿ',
        'Compilation Overhead:\nGates ∼ 4ⁿ',
        'QEC Data Scarcity:\nSamples ≪ 10⁵'
    ]
    
    for i, (y, text) in enumerate(zip(problem_y, problem_texts)):
        box = FancyBboxPatch((1, y-0.5), 2.5, 1.0,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['problem'],
                            alpha=0.3,
                            edgecolor=colors['problem'],
                            linewidth=0.5)
        ax.add_patch(box)
        ax.text(2.25, y, text, ha='center', va='center',
                fontsize=7, fontweight='bold', color=colors['problem'])
    
    # Solution boxes
    solution_y = [7, 4.5, 2]
    solution_texts = [
        'Noise-Aware\nInitialization',
        'Hardware-Optimized\nCompilation',
        'Physics-Informed\nSynthetic Data'
    ]
    
    for i, (y, text) in enumerate(zip(solution_y, solution_texts)):
        box = FancyBboxPatch((4, y-0.5), 2.5, 1.0,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['solution'],
                            alpha=0.3,
                            edgecolor=colors['solution'],
                            linewidth=0.5)
        ax.add_patch(box)
        ax.text(5.25, y, text, ha='center', va='center',
                fontsize=7, fontweight='bold', color=colors['solution'])
    
    # Result box
    result_box = FancyBboxPatch((7, 3.5), 2.5, 3.0,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['result'],
                               alpha=0.3,
                               edgecolor=colors['result'],
                               linewidth=0.5)
    ax.add_patch(result_box)
    ax.text(8.25, 5.5, 'Integrated\nQuantum\nAlgorithm', ha='center', va='center',
            fontsize=8, fontweight='bold', color=colors['result'])
    ax.text(8.25, 4.0, '40–50×\nImprovement', ha='center', va='center',
            fontsize=7, color=colors['result'], fontweight='bold')
    
    # Arrows from problems to solutions
    for y in problem_y:
        ax.annotate('', xy=(4, y), xytext=(3.5, y),
                   arrowprops=dict(arrowstyle='->',
                                   color=colors['arrow'],
                                   lw=1,
                                   shrinkA=5,
                                   shrinkB=5))
    
    # Arrows from solutions to result
    for y in solution_y:
        ax.annotate('', xy=(7, y), xytext=(6.5, y),
                   arrowprops=dict(arrowstyle='->',
                                   color=colors['arrow'],
                                   lw=1,
                                   shrinkA=5,
                                   shrinkB=5))
    
    # Title
    ax.text(5, 9.5, 'AdaptiveQuantum Framework', ha='center', va='center',
            fontsize=10, fontweight='bold')
    
    # Subtitle
    ax.text(5, 8.8, 'ML-Driven Quantum Co-Design', ha='center', va='center',
            fontsize=8, style='italic')
    
    # Legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=colors['problem'], alpha=0.3, label='Quantum Bottlenecks'),
        plt.Rectangle((0,0),1,1, facecolor=colors['solution'], alpha=0.3, label='ML Solutions'),
        plt.Rectangle((0,0),1,1, facecolor=colors['result'], alpha=0.3, label='Integrated Result')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.15),
              fontsize=6, frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path("figures/nature_physics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / "fig1_conceptual_framework.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig1_conceptual_framework.pdf", bbox_inches='tight')
    plt.savefig(output_dir / "fig1_conceptual_framework.svg", bbox_inches='tight')
    
    print(f"✓ Figure 1 saved to {output_dir}/")
    plt.show()
    
    return True

if __name__ == "__main__":
    create_conceptual_framework()
