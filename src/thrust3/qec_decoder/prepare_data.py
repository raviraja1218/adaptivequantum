"""
Prepare QEC training dataset with multiple noise types.
"""
import torch
import numpy as np
import pickle
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.thrust3.qec_decoder.surface_code import SurfaceCode
import yaml

def prepare_qec_dataset():
    """Generate QEC dataset with multiple noise types."""
    print("🔄 Preparing QEC dataset...")
    
    # Load configuration
    with open('config/phase4_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize surface code
    sc = SurfaceCode(distance=config['surface_code']['distance'])
    
    datasets = {}
    
    # Generate data for each noise type
    for noise_type in config['surface_code']['noise_types']:
        print(f"  Generating {noise_type} noise data...")
        
        n_samples = config['data']['real_samples'] // len(config['surface_code']['noise_types'])
        
        dataset = sc.generate_dataset(
            n_samples=n_samples,
            noise_type=noise_type
        )
        
        datasets[noise_type] = dataset
        
        print(f"    Generated {n_samples} samples")
        print(f"    Error rate: {dataset['errors'].float().mean():.4f}")
    
    # Combine datasets
    all_errors = []
    all_syndromes = []
    noise_labels = []
    
    for noise_type, dataset in datasets.items():
        all_errors.append(dataset['errors'])
        all_syndromes.append(dataset['syndromes'])
        noise_labels.extend([noise_type] * len(dataset['errors']))
    
    combined_dataset = {
        'errors': torch.cat(all_errors, dim=0),
        'syndromes': torch.cat(all_syndromes, dim=0),
        'noise_labels': noise_labels,
        'n_total': len(all_errors[0]) * len(datasets)
    }
    
    # Split into train/val/test
    n_total = combined_dataset['n_total']
    train_split = config['data']['train_split']
    val_split = config['data']['val_split']
    test_split = config['data']['test_split']
    
    indices = torch.randperm(n_total)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    # Create splits
    splits = {
        'train': {
            'errors': combined_dataset['errors'][train_idx],
            'syndromes': combined_dataset['syndromes'][train_idx],
            'noise_labels': [combined_dataset['noise_labels'][i] for i in train_idx]
        },
        'val': {
            'errors': combined_dataset['errors'][val_idx],
            'syndromes': combined_dataset['syndromes'][val_idx],
            'noise_labels': [combined_dataset['noise_labels'][i] for i in val_idx]
        },
        'test': {
            'errors': combined_dataset['errors'][test_idx],
            'syndromes': combined_dataset['syndromes'][test_idx],
            'noise_labels': [combined_dataset['noise_labels'][i] for i in test_idx]
        }
    }
    
    # Save dataset
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "qec_dataset.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(splits, f)
    
    # Generate statistics
    stats = {
        'total_samples': n_total,
        'train_samples': len(train_idx),
        'val_samples': len(val_idx),
        'test_samples': len(test_idx),
        'noise_distribution': {
            noise_type: sum(1 for label in combined_dataset['noise_labels'] if label == noise_type)
            for noise_type in config['surface_code']['noise_types']
        },
        'error_rate': combined_dataset['errors'].float().mean().item()
    }
    
    stats_path = Path("experiments/thrust3/dataset_statistics.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Dataset prepared successfully!")
    print(f"   Saved to: {output_path}")
    print(f"   Total samples: {n_total}")
    print(f"   Train/Val/Test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    
    # Create quick visualization
    create_visualization(combined_dataset, splits)
    
    return splits, stats

def create_visualization(dataset, splits):
    """Create initial visualization of error distribution."""
    import matplotlib.pyplot as plt
    from collections import Counter
    
    # Count noise types
    noise_counts = Counter(dataset['noise_labels'])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Noise type distribution
    labels = list(noise_counts.keys())
    counts = list(noise_counts.values())
    
    axes[0].bar(labels, counts)
    axes[0].set_title('Noise Type Distribution')
    axes[0].set_ylabel('Number of Samples')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        axes[0].text(i, count + max(counts)*0.02, str(count), 
                    ha='center', va='bottom')
    
    # Plot 2: Error weight distribution
    error_weights = dataset['errors'].sum(dim=1).numpy()
    
    axes[1].hist(error_weights, bins=range(int(error_weights.max()) + 2), 
                edgecolor='black', alpha=0.7)
    axes[1].set_title('Error Weight Distribution')
    axes[1].set_xlabel('Number of Errors')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    # Save figure
    fig_dir = Path("figures/exploratory/thrust3")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    fig_path = fig_dir / "error_distribution_initial.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Visualization saved to: {fig_path}")

if __name__ == "__main__":
    dataset, stats = prepare_qec_dataset()
