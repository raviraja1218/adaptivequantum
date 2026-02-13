"""
Fixed synthetic data generation script.
"""
import torch
import pickle
import numpy as np
from pathlib import Path
import sys
import yaml
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.thrust3.vae_error_generator.model.conditional_vae_fixed import (
    ConditionalVAEFixed, 
    NoiseParameterEncoderFixed
)

def generate_synthetic_data_fixed(config_path: str = 'config/phase4_config.yaml'):
    """Generate synthetic error data using fixed VAE."""
    print("🔄 Generating synthetic error data (fixed)...")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    n_samples = config['data']['synthetic_samples']
    
    # Load device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load real dataset to get dimensions
    with open('data/processed/qec_dataset.pkl', 'rb') as f:
        real_dataset = pickle.load(f)
    
    error_dim = real_dataset['train']['errors'].shape[1]
    
    # Load trained VAE
    model_path = Path("models/saved/conditional_vae_fixed.pt")
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None, None
    
    checkpoint = torch.load(model_path, map_location=device)
    
    vae = ConditionalVAEFixed(
        error_dim=error_dim,
        latent_dim=checkpoint['latent_dim'],
        noise_param_dim=4,
        hidden_dims=checkpoint['config']['hidden_dims']
    )
    
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae = vae.to(device)
    vae.eval()
    
    print(f"✅ Loaded VAE from: {model_path}")
    print(f"   Error dimension: {error_dim}")
    print(f"   Latent dimension: {checkpoint['latent_dim']}")
    print(f"   Generating {n_samples:,} synthetic samples")
    
    # Initialize noise encoder
    noise_encoder = NoiseParameterEncoderFixed()
    
    # Generate synthetic data in batches
    batch_size = 1000
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    all_synthetic_errors = []
    all_noise_params = []
    all_noise_types = []
    
    noise_types = config['surface_code']['noise_types']
    noise_probs = [0.4, 0.2, 0.2, 0.2]  # Higher probability for depolarizing
    
    for batch_idx in tqdm(range(n_batches), desc="Generating batches"):
        # Determine samples for this batch
        current_batch_size = min(batch_size, n_samples - batch_idx * batch_size)
        
        # Sample noise types
        batch_noise_types = np.random.choice(noise_types, current_batch_size, p=noise_probs)
        
        # Encode noise parameters
        batch_noise_params = []
        for noise_type in batch_noise_types:
            params = noise_encoder.encode(noise_type)
            batch_noise_params.append(params.numpy())
        
        batch_noise_params = torch.tensor(np.array(batch_noise_params), dtype=torch.float32).to(device)
        
        # Generate synthetic errors
        with torch.no_grad():
            synthetic_errors, error_probs = vae.sample(
                current_batch_size, 
                batch_noise_params, 
                device
            )
        
        # Store results
        all_synthetic_errors.append(synthetic_errors.cpu())
        all_noise_params.append(batch_noise_params.cpu())
        all_noise_types.extend(batch_noise_types)
    
    # Combine all batches
    synthetic_errors = torch.cat(all_synthetic_errors, dim=0)
    noise_params = torch.cat(all_noise_params, dim=0)
    
    print(f"\n✅ Generated {len(synthetic_errors):,} synthetic samples")
    print(f"   Error rate: {synthetic_errors.float().mean():.4f}")
    
    # Create synthetic dataset
    synthetic_dataset = {
        'errors': synthetic_errors,
        'noise_params': noise_params,
        'noise_types': all_noise_types,
        'n_samples': len(synthetic_errors),
        'generation_method': 'conditional_vae_fixed',
        'error_dim': error_dim
    }
    
    # Save synthetic data
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "synthetic_errors_fixed.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(synthetic_dataset, f)
    
    print(f"✅ Synthetic data saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Generate statistics and quality metrics
    stats = compute_quality_metrics(synthetic_dataset, real_dataset)
    
    return synthetic_dataset, stats

def compute_quality_metrics(synthetic_data, real_data):
    """Compute quality metrics for synthetic data."""
    print("\n📊 Computing quality metrics...")
    
    # Extract data
    real_errors = real_data['train']['errors']
    synth_errors = synthetic_data['errors']
    
    # Basic statistics
    real_error_rate = real_errors.float().mean().item()
    synth_error_rate = synth_errors.float().mean().item()
    error_rate_diff = abs(real_error_rate - synth_error_rate)
    
    # Compute KL divergence for Bernoulli distributions
    eps = 1e-10
    p_real = max(eps, min(1-eps, real_error_rate))
    p_synth = max(eps, min(1-eps, synth_error_rate))
    
    kl_divergence = p_real * np.log(p_real / p_synth) + \
                   (1 - p_real) * np.log((1 - p_real) / (1 - p_synth))
    
    # Compute per-qubit error rates
    real_qubit_rates = real_errors.float().mean(dim=0).numpy()
    synth_qubit_rates = synth_errors.float().mean(dim=0).numpy()
    
    # Compute correlation
    correlation = np.corrcoef(real_qubit_rates, synth_qubit_rates)[0, 1]
    
    # Compute histogram distance for error weights
    real_weights = real_errors.sum(dim=1).float().numpy()
    synth_weights = synth_errors.sum(dim=1).float().numpy()
    
    max_weight = int(max(real_weights.max(), synth_weights.max()))
    bins = range(max_weight + 2)
    
    real_hist, _ = np.histogram(real_weights, bins=bins, density=True)
    synth_hist, _ = np.histogram(synth_weights, bins=bins, density=True)
    
    hist_distance = np.abs(real_hist - synth_hist).mean()
    
    # Create statistics dictionary
    stats = {
        'real_error_rate': float(real_error_rate),
        'synthetic_error_rate': float(synth_error_rate),
        'error_rate_difference': float(error_rate_diff),
        'kl_divergence': float(kl_divergence),
        'per_qubit_correlation': float(correlation),
        'histogram_distance': float(hist_distance),
        'n_synthetic_samples': len(synth_errors),
        'quality_assessment': 'good' if kl_divergence < 0.05 else 'needs_improvement',
        'paper_target_met': kl_divergence < 0.05
    }
    
    print(f"  Real error rate: {real_error_rate:.4f}")
    print(f"  Synthetic error rate: {synth_error_rate:.4f}")
    print(f"  Error rate difference: {error_rate_diff:.6f}")
    print(f"  KL Divergence: {kl_divergence:.6f}")
    print(f"  Per-qubit correlation: {correlation:.4f}")
    print(f"  Histogram distance: {hist_distance:.6f}")
    print(f"  Quality assessment: {stats['quality_assessment'].upper()}")
    
    # Save statistics
    stats_dir = Path("experiments/thrust3/synthetic_validation")
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    stats_path = stats_dir / "synthetic_quality_metrics_fixed.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  Quality metrics saved to: {stats_path}")
    
    # Generate visualization
    generate_quality_visualization(real_data, synthetic_data, stats, stats_dir)
    
    return stats

def generate_quality_visualization(real_data, synthetic_data, stats, output_dir):
    """Generate visualization comparing real and synthetic data."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract data
    real_errors = real_data['train']['errors']
    synth_errors = synthetic_data['errors']
    
    # Plot 1: Error weight distribution
    real_weights = real_errors.sum(dim=1).float().numpy()
    synth_weights = synth_errors.sum(dim=1).float().numpy()
    
    max_weight = int(max(real_weights.max(), synth_weights.max()))
    bins = range(max_weight + 2)
    
    axes[0, 0].hist(real_weights, bins=bins, alpha=0.7, label='Real', 
                   edgecolor='black', density=True)
    axes[0, 0].hist(synth_weights, bins=bins, alpha=0.7, label='Synthetic', 
                   edgecolor='black', density=True)
    axes[0, 0].set_xlabel('Error Weight (# of errors)')
    axes[0, 0].set_ylabel('Probability Density')
    axes[0, 0].set_title('Error Weight Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Error rate per qubit position
    real_qubit_rates = real_errors.float().mean(dim=0).numpy()
    synth_qubit_rates = synth_errors.float().mean(dim=0).numpy()
    
    x = range(len(real_qubit_rates))
    axes[0, 1].bar([xi - 0.2 for xi in x], real_qubit_rates, width=0.4, 
                   label='Real', alpha=0.8)
    axes[0, 1].bar([xi + 0.2 for xi in x], synth_qubit_rates, width=0.4, 
                   label='Synthetic', alpha=0.8)
    axes[0, 1].set_xlabel('Qubit Position')
    axes[0, 1].set_ylabel('Error Probability')
    axes[0, 1].set_title('Error Distribution by Qubit')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Noise type distribution
    from collections import Counter
    real_noise_counts = Counter(real_data['train']['noise_labels'])
    synth_noise_counts = Counter(synthetic_data['noise_types'])
    
    noise_types = list(real_noise_counts.keys())
    real_counts = [real_noise_counts[nt] for nt in noise_types]
    synth_counts = [synth_noise_counts[nt] for nt in noise_types]
    
    x = range(len(noise_types))
    axes[1, 0].bar([xi - 0.2 for xi in x], real_counts, width=0.4, 
                   label='Real', alpha=0.8)
    axes[1, 0].bar([xi + 0.2 for xi in x], synth_counts, width=0.4, 
                   label='Synthetic', alpha=0.8)
    axes[1, 0].set_xlabel('Noise Type')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Noise Type Distribution')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(noise_types, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Correlation plot
    axes[1, 1].scatter(real_qubit_rates, synth_qubit_rates, alpha=0.6, s=50)
    
    # Add correlation line
    z = np.polyfit(real_qubit_rates, synth_qubit_rates, 1)
    p = np.poly1d(z)
    x_range = np.linspace(real_qubit_rates.min(), real_qubit_rates.max(), 100)
    axes[1, 1].plot(x_range, p(x_range), 'r--', alpha=0.8, 
                   label=f'ρ = {stats["per_qubit_correlation"]:.3f}')
    
    # Add ideal line
    max_val = max(real_qubit_rates.max(), synth_qubit_rates.max())
    axes[1, 1].plot([0, max_val], [0, max_val], 'g:', alpha=0.6, label='Ideal')
    
    axes[1, 1].set_xlabel('Real Error Probability')
    axes[1, 1].set_ylabel('Synthetic Error Probability')
    axes[1, 1].set_title(f'Per-Qubit Correlation (ρ = {stats["per_qubit_correlation"]:.3f})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add KL divergence annotation
    plt.figtext(0.02, 0.02, f'KL Divergence: {stats["kl_divergence"]:.6f}', 
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    fig_path = output_dir / "synthetic_quality_comparison_fixed.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Quality visualization saved to: {fig_path}")

if __name__ == "__main__":
    synthetic_data, stats = generate_synthetic_data_fixed()
    
    if synthetic_data is not None:
        # Check against paper targets
        print("\n🎯 PAPER TARGET VERIFICATION:")
        print("=" * 40)
        
        if stats['paper_target_met']:
            print(f"✅ KL divergence: {stats['kl_divergence']:.6f} < 0.05")
            print("   Synthetic errors are statistically similar to real errors!")
        else:
            print(f"❌ KL divergence: {stats['kl_divergence']:.6f} ≥ 0.05")
            print("   Synthetic errors need improvement for paper submission.")
        
        if stats['error_rate_difference'] < 0.01:
            print(f"✅ Error rate difference: {stats['error_rate_difference']:.6f} < 0.01")
        else:
            print(f"⚠️  Error rate difference: {stats['error_rate_difference']:.6f} ≥ 0.01")
        
        if stats['per_qubit_correlation'] > 0.8:
            print(f"✅ Per-qubit correlation: {stats['per_qubit_correlation']:.3f} > 0.8")
        else:
            print(f"⚠️  Per-qubit correlation: {stats['per_qubit_correlation']:.3f} ≤ 0.8")
        
        print("=" * 40)
        
        # Overall assessment
        if stats['paper_target_met']:
            print("🎉 SYNTHETIC DATA QUALITY: EXCELLENT (Ready for paper)")
        else:
            print("⚠️  SYNTHETIC DATA QUALITY: NEEDS IMPROVEMENT")
