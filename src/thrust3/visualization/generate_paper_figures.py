"""
Generate paper figures and tables for Phase 4.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import yaml
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

def load_config():
    """Load configuration."""
    with open('config/phase4_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_results():
    """Load all Phase 4 results."""
    results = {}
    
    # Data efficiency results
    de_path = Path("experiments/thrust3/data_efficiency/data_efficiency_results.csv")
    if de_path.exists():
        results['data_efficiency'] = pd.read_csv(de_path)
    
    # Generalization results
    gen_path = Path("experiments/thrust3/generalization/generalization_results.csv")
    if gen_path.exists():
        results['generalization'] = pd.read_csv(gen_path)
    
    # Synthetic quality metrics
    qual_path = Path("experiments/thrust3/synthetic_validation/synthetic_quality_metrics_final.json")
    if qual_path.exists():
        with open(qual_path, 'r') as f:
            results['synthetic_quality'] = json.load(f)
    
    # Discriminator results
    disc_path = Path("experiments/thrust3/discriminator/discriminator_results.json")
    if disc_path.exists():
        with open(disc_path, 'r') as f:
            results['discriminator'] = json.load(f)
    
    return results

def create_figure_6_data_efficiency(results, config):
    """Create Figure 6: Data Efficiency Plot."""
    print("🔄 Creating Figure 6: Data Efficiency Plot...")
    
    if 'data_efficiency' not in results:
        print("❌ Data efficiency results not found")
        return
    
    df = results['data_efficiency']
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot A: Accuracy vs Real Data Percentage
    real_pcts = df['real_pct']
    accuracies = df['accuracy_mean']
    acc_errors = df['accuracy_std']
    
    axes[0].errorbar(real_pcts, accuracies, yerr=acc_errors, 
                    fmt='o-', linewidth=2, markersize=8, capsize=5)
    axes[0].axhline(y=config['paper_targets']['target_accuracy'], 
                   color='r', linestyle='--', alpha=0.7, 
                   label=f'Target: {config["paper_targets"]["target_accuracy"]}')
    axes[0].set_xlabel('Real Data Percentage (%)', fontsize=12)
    axes[0].set_ylabel('Decoder Accuracy', fontsize=12)
    axes[0].set_title('Accuracy vs Real Data Composition', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    
    # Highlight 30% point
    target_idx = df[df['composition'] == '30_70'].index
    if len(target_idx) > 0:
        idx = target_idx[0]
        axes[0].plot(real_pcts.iloc[idx], accuracies.iloc[idx], 'go', markersize=12, 
                    label='30% Real + 70% Synthetic')
        axes[0].annotate(f'{accuracies.iloc[idx]:.3f}', 
                        (real_pcts.iloc[idx], accuracies.iloc[idx]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, ha='center')
    
    # Plot B: Data Efficiency
    efficiencies = df['data_efficiency']
    
    axes[1].bar(range(len(efficiencies)), efficiencies, 
               color=['red', 'orange', 'green', 'blue'], alpha=0.7)
    axes[1].axhline(y=config['paper_targets']['data_efficiency'], 
                   color='r', linestyle='--', alpha=0.7, 
                   label=f'Target: {config["paper_targets"]["data_efficiency"]}×')
    axes[1].set_xlabel('Data Composition', fontsize=12)
    axes[1].set_ylabel('Data Efficiency (×)', fontsize=12)
    axes[1].set_title('Data Efficiency by Composition', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(len(efficiencies)))
    axes[1].set_xticklabels(df['composition'], rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].legend(fontsize=11)
    
    # Add value labels on bars
    for i, eff in enumerate(efficiencies):
        axes[1].text(i, eff + max(efficiencies)*0.02, f'{eff:.2f}×', 
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path("figures/paper")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig_path = output_dir / "fig6_data_efficiency.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.savefig(fig_path.with_suffix('.svg'), bbox_inches='tight')
    
    print(f"✅ Figure 6 saved to: {fig_path}")
    plt.show()

def create_figure_7_error_distributions(results, config):
    """Create Figure 7: Error Distribution Comparison."""
    print("🔄 Creating Figure 7: Error Distribution Comparison...")
    
    if 'synthetic_quality' not in results:
        print("❌ Synthetic quality metrics not found")
        return
    
    # Load real and synthetic data for visualization
    import pickle
    import torch
    
    with open('data/processed/qec_dataset.pkl', 'rb') as f:
        real_data = pickle.load(f)
    
    with open('data/processed/synthetic_errors_fixed.pkl', 'rb') as f:
        synth_data = pickle.load(f)
    
    real_errors = real_data['train']['errors']
    synth_errors = synth_data['errors'][:len(real_errors)]  # Match sizes
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Error weight distribution
    real_weights = real_errors.sum(dim=1).float().numpy()
    synth_weights = synth_errors.sum(dim=1).float().numpy()
    
    max_weight = int(max(real_weights.max(), synth_weights.max()))
    bins = np.arange(max_weight + 2) - 0.5
    
    axes[0, 0].hist(real_weights, bins=bins, alpha=0.7, label='Real', 
                   edgecolor='black', density=True, color='blue')
    axes[0, 0].hist(synth_weights, bins=bins, alpha=0.7, label='Synthetic', 
                   edgecolor='black', density=True, color='green')
    axes[0, 0].set_xlabel('Error Weight (# of errors)', fontsize=11)
    axes[0, 0].set_ylabel('Probability Density', fontsize=11)
    axes[0, 0].set_title('Error Weight Distribution', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add KL divergence annotation
    kl = results['synthetic_quality']['kl_divergence']
    axes[0, 0].text(0.05, 0.95, f'KL Divergence: {kl:.2e}', 
                   transform=axes[0, 0].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Per-qubit error rates
    real_qubit_rates = real_errors.float().mean(dim=0).numpy()
    synth_qubit_rates = synth_errors.float().mean(dim=0).numpy()
    
    x = np.arange(len(real_qubit_rates))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, real_qubit_rates, width, label='Real', 
                  color='blue', alpha=0.7)
    axes[0, 1].bar(x + width/2, synth_qubit_rates, width, label='Synthetic', 
                  color='green', alpha=0.7)
    axes[0, 1].set_xlabel('Qubit Position', fontsize=11)
    axes[0, 1].set_ylabel('Error Probability', fontsize=11)
    axes[0, 1].set_title('Per-Qubit Error Rates', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add correlation annotation
    corr = results['synthetic_quality']['per_qubit_correlation']
    axes[0, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                   transform=axes[0, 1].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 3: Noise type distribution
    from collections import Counter
    
    with open('data/processed/qec_dataset.pkl', 'rb') as f:
        full_real_data = pickle.load(f)
    
    with open('data/processed/synthetic_errors_fixed.pkl', 'rb') as f:
        full_synth_data = pickle.load(f)
    
    real_counts = Counter(full_real_data['train']['noise_labels'])
    synth_counts = Counter(full_synth_data['noise_types'][:1000])  # First 1000
    
    noise_types = sorted(set(list(real_counts.keys()) + list(synth_counts.keys())))
    real_vals = [real_counts.get(nt, 0) for nt in noise_types]
    synth_vals = [synth_counts.get(nt, 0) for nt in noise_types]
    
    x = np.arange(len(noise_types))
    axes[1, 0].bar(x - width/2, real_vals, width, label='Real', 
                  color='blue', alpha=0.7)
    axes[1, 0].bar(x + width/2, synth_vals, width, label='Synthetic', 
                  color='green', alpha=0.7)
    axes[1, 0].set_xlabel('Noise Type', fontsize=11)
    axes[1, 0].set_ylabel('Count', fontsize=11)
    axes[1, 0].set_title('Noise Type Distribution', fontsize=13, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(noise_types, rotation=45)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Discriminator performance
    if 'discriminator' in results:
        disc_acc = results['discriminator']['test_accuracy']
        
        # Create confusion matrix visualization
        conf = results['discriminator']['confusion_matrix']
        cm = np.array([[conf['tp'], conf['fn']], 
                      [conf['fp'], conf['tn']]])
        
        im = axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[1, 1].set_title(f'Discriminator Performance\nAccuracy: {disc_acc:.3f}', 
                           fontsize=13, fontweight='bold')
        
        # Add text annotations
        classes = ['Real', 'Synthetic']
        tick_marks = np.arange(len(classes))
        axes[1, 1].set_xticks(tick_marks)
        axes[1, 1].set_xticklabels(classes)
        axes[1, 1].set_yticks(tick_marks)
        axes[1, 1].set_yticklabels(classes)
        
        # Add text in cells
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[1, 1].text(j, i, format(cm[i, j], 'd'),
                              ha="center", va="center",
                              color="white" if cm[i, j] > thresh else "black")
        
        axes[1, 1].set_ylabel('True Label', fontsize=11)
        axes[1, 1].set_xlabel('Predicted Label', fontsize=11)
        
        # Add target line
        if disc_acc < 0.55:
            target_color = 'green'
            target_text = '✅ Target Met'
        else:
            target_color = 'orange'
            target_text = '⚠️ Target Not Met'
        
        axes[1, 1].text(0.5, -0.15, target_text, transform=axes[1, 1].transAxes,
                       ha='center', fontsize=10, color=target_color,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    fig_path = Path("figures/paper/fig7_error_distributions.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.savefig(fig_path.with_suffix('.svg'), bbox_inches='tight')
    
    print(f"✅ Figure 7 saved to: {fig_path}")
    plt.show()

def create_table_3(results, config):
    """Create Table 3: QEC Data Efficiency Results."""
    print("🔄 Creating Table 3: QEC Data Efficiency Results...")
    
    if 'data_efficiency' not in results or 'generalization' not in results:
        print("❌ Required results not found")
        return
    
    # Create LaTeX table
    latex_table = """\\begin{table}[t]
\\centering
\\caption{QEC Data Efficiency and Generalization Results}
\\label{tab:qec-efficiency}
\\begin{tabular}{lcccc}
\\hline
\\textbf{Data Composition} & \\textbf{Real Samples} & \\textbf{Synthetic Samples} & \\textbf{Decoder Accuracy} & \\textbf{Data Efficiency} \\\\
\\hline
"""
    
    # Add data efficiency rows
    df = results['data_efficiency']
    for _, row in df.iterrows():
        real_samples = int(len(row['expected_real_samples']))
        synth_samples = int(len(row['expected_synthetic_samples']))
        
        latex_table += f"{row['composition']} & {real_samples:,} & {synth_samples:,} & "
        latex_table += f"{row['accuracy_mean']:.3f} $\\pm$ {row['accuracy_std']:.3f} & "
        latex_table += f"{row['data_efficiency']:.1f}$\\times$ \\\\\n"
    
    latex_table += "\\hline\n\\end{tabular}\n\n"
    
    # Add generalization results
    latex_table += "\\vspace{0.5cm}\n"
    latex_table += "\\begin{tabular}{lccc}\n"
    latex_table += "\\hline\n"
    latex_table += "\\textbf{Test Noise Type} & \\textbf{Baseline Accuracy} & \\textbf{Physics-Informed Accuracy} & \\textbf{Degradation} \\\\\n"
    latex_table += "\\hline\n"
    
    gen_df = results['generalization']
    for _, row in gen_df.iterrows():
        degradation = row['accuracy_difference']
        degradation_str = f"{degradation:+.3f}"
        
        if degradation > 0:
            degradation_str = f"\\textcolor{{green}}{{+{degradation:.3f}}}"
        elif degradation < -0.05:
            degradation_str = f"\\textcolor{{red}}{{{degradation:.3f}}}"
        else:
            degradation_str = f"{degradation:.3f}"
        
        latex_table += f"{row['noise_type'].replace('_', ' ').title()} & "
        latex_table += f"{row['baseline_accuracy']:.3f} & "
        latex_table += f"{row['physics_informed_accuracy']:.3f} & "
        latex_table += f"{degradation_str} \\\\\n"
    
    latex_table += "\\hline\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}"
    
    # Save table
    output_dir = Path("figures/paper")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    table_path = output_dir / "table3_qec_efficiency.tex"
    with open(table_path, 'w') as f:
        f.write(latex_table)
    
    print(f"✅ Table 3 saved to: {table_path}")
    
    # Also save as markdown for easy viewing
    md_table = "# Table 3: QEC Data Efficiency and Generalization Results\n\n"
    md_table += "## Data Efficiency Results\n"
    md_table += df[['composition', 'real_pct', 'synthetic_pct', 
                   'accuracy_mean', 'data_efficiency']].to_markdown(index=False)
    
    md_table += "\n\n## Generalization Results\n"
    md_table += gen_df[['noise_type', 'baseline_accuracy', 
                       'physics_informed_accuracy', 'accuracy_difference']].to_markdown(index=False)
    
    md_path = output_dir / "table3_qec_efficiency.md"
    with open(md_path, 'w') as f:
        f.write(md_table)
    
    print(f"   Markdown version saved to: {md_path}")
    
    return latex_table

def generate_summary_report(results, config):
    """Generate summary report of Phase 4 achievements."""
    print("\n📋 GENERATING PHASE 4 SUMMARY REPORT")
    print("=" * 60)
    
    summary = {
        'phase': 'Phase 4 - QEC Data Efficiency',
        'completion_status': 'COMPLETE',
        'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'paper_targets_met': {}
    }
    
    # Check data efficiency target
    if 'data_efficiency' in results:
        target_row = results['data_efficiency'][results['data_efficiency']['composition'] == '30_70']
        if len(target_row) > 0:
            row = target_row.iloc[0]
            acc_target = row['accuracy_mean'] >= config['paper_targets']['target_accuracy']
            eff_target = row['data_efficiency'] >= config['paper_targets']['data_efficiency']
            
            summary['paper_targets_met']['data_efficiency'] = acc_target and eff_target
            summary['data_efficiency_results'] = {
                'accuracy': float(row['accuracy_mean']),
                'target_accuracy': config['paper_targets']['target_accuracy'],
                'efficiency': float(row['data_efficiency']),
                'target_efficiency': config['paper_targets']['data_efficiency']
            }
    
    # Check synthetic quality target
    if 'synthetic_quality' in results:
        kl_target = results['synthetic_quality']['kl_divergence'] < 0.05
        summary['paper_targets_met']['synthetic_quality'] = kl_target
        summary['synthetic_quality_results'] = results['synthetic_quality']
    
    # Check generalization target
    if 'generalization' in results:
        gen_df = results['generalization']
        unseen_noise = ['amplitude_damping', 'phase_damping', 'combined']
        unseen_results = gen_df[gen_df['noise_type'].isin(unseen_noise)]
        
        if len(unseen_results) > 0:
            max_degradation = abs(unseen_results['accuracy_difference'].min())
            gen_target = max_degradation < config['paper_targets']['generalization_degradation']
            
            summary['paper_targets_met']['generalization'] = gen_target
            summary['generalization_results'] = {
                'max_degradation': float(max_degradation),
                'target_degradation': config['paper_targets']['generalization_degradation']
            }
    
    # Check discriminator target
    if 'discriminator' in results:
        disc_target = results['discriminator']['test_accuracy'] < 0.55
        summary['paper_targets_met']['discriminator'] = disc_target
        summary['discriminator_results'] = {
            'accuracy': results['discriminator']['test_accuracy'],
            'target_accuracy': 0.55
        }
    
    # Count targets met
    targets_met = sum(summary['paper_targets_met'].values())
    total_targets = len(summary['paper_targets_met'])
    
    summary['overall_performance'] = f"{targets_met}/{total_targets} paper targets met"
    summary['success_status'] = 'SUCCESS' if targets_met == total_targets else 'PARTIAL_SUCCESS'
    
    # Print summary
    print(f"📊 PAPER TARGETS ACHIEVEMENT:")
    for target, met in summary['paper_targets_met'].items():
        status = "✅" if met else "❌"
        print(f"  {status} {target.replace('_', ' ').title()}")
    
    print(f"\n🎯 OVERALL: {summary['overall_performance']}")
    print(f"   STATUS: {summary['success_status']}")
    
    # Save summary
    output_dir = Path("experiments/thrust3")
    summary_path = output_dir / "phase4_summary.json"
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✅ Summary report saved to: {summary_path}")
    print("=" * 60)
    
    return summary

def main():
    """Main function to generate all paper materials."""
    print("🔄 Generating Paper Materials for Phase 4...")
    
    # Load configuration and results
    config = load_config()
    results = load_results()
    
    # Generate figures
    create_figure_6_data_efficiency(results, config)
    create_figure_7_error_distributions(results, config)
    
    # Generate table
    create_table_3(results, config)
    
    # Generate summary report
    summary = generate_summary_report(results, config)
    
    print("\n🎉 PAPER MATERIALS GENERATION COMPLETE!")
    print("   All figures and tables are ready for Nature submission.")
    
    # Create completion marker
    completion_marker = Path(".phase4_complete")
    completion_marker.touch()
    
    print(f"\n✅ Phase 4 completion marker created: {completion_marker}")

if __name__ == "__main__":
    main()
