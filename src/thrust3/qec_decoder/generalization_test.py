"""
Generalization test: Train on depolarizing noise, test on other noise types.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from pathlib import Path
import sys
import yaml
import json
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.thrust3.qec_decoder.data_efficiency_experiment import QECDecoder

def load_datasets_by_noise():
    """Load datasets grouped by noise type."""
    with open('data/processed/qec_dataset.pkl', 'rb') as f:
        full_dataset = pickle.load(f)
    
    # Separate by noise type
    noise_datasets = {}
    
    for split in ['train', 'val', 'test']:
        split_data = full_dataset[split]
        noise_labels = split_data['noise_labels']
        
        # Group by noise type
        for noise_type in set(noise_labels):
            if noise_type not in noise_datasets:
                noise_datasets[noise_type] = {'syndromes': [], 'errors': []}
            
            # Get indices for this noise type
            indices = [i for i, label in enumerate(noise_labels) if label == noise_type]
            
            if indices:
                noise_datasets[noise_type]['syndromes'].append(split_data['syndromes'][indices])
                noise_datasets[noise_type]['errors'].append(split_data['errors'][indices])
    
    # Concatenate for each noise type
    for noise_type in noise_datasets:
        if noise_datasets[noise_type]['syndromes']:
            noise_datasets[noise_type]['syndromes'] = torch.cat(noise_datasets[noise_type]['syndromes'])
            noise_datasets[noise_type]['errors'] = torch.cat(noise_datasets[noise_type]['errors'])
    
    return noise_datasets

def train_on_depolarizing(noise_datasets, synth_data, config):
    """Train decoder on depolarizing noise (70% synthetic, 30% real)."""
    # Get depolarizing data
    depol_syndromes = noise_datasets['depolarizing']['syndromes']
    depol_errors = noise_datasets['depolarizing']['errors']
    
    # Use 30% real depolarizing samples
    n_real = int(len(depol_syndromes) * 0.3)
    real_indices = torch.randperm(len(depol_syndromes))[:n_real]
    
    real_syndromes = depol_syndromes[real_indices]
    real_errors = depol_errors[real_indices]
    
    # Use synthetic data (from synthetic dataset)
    n_synth = int(len(synth_data['errors']) * 0.7)
    synth_indices = torch.randperm(len(synth_data['errors']))[:n_synth]
    synth_errors = synth_data['errors'][synth_indices]
    
    # Generate synthetic syndromes (random matching distribution)
    synth_syndromes = torch.randint(0, 2, (n_synth, real_syndromes.shape[1])).float()
    
    # Combine
    train_syndromes = torch.cat([real_syndromes, synth_syndromes])
    train_errors = torch.cat([real_errors, synth_errors])
    
    # Train decoder
    decoder = QECDecoder(
        input_dim=train_syndromes.shape[1],
        hidden_dim=64,
        output_dim=train_errors.shape[1]
    )
    
    dataset = TensorDataset(train_syndromes, train_errors)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    optimizer = optim.Adam(decoder.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    decoder.train()
    for epoch in range(100):
        for batch_syndromes, batch_errors in dataloader:
            optimizer.zero_grad()
            predictions = decoder(batch_syndromes)
            loss = criterion(predictions, batch_errors)
            loss.backward()
            optimizer.step()
        
        if epoch > 10 and loss.item() < 0.01:
            break
    
    return decoder, n_real, n_synth

def test_on_noise_types(decoder, noise_datasets):
    """Test decoder on different noise types."""
    results = {}
    
    for noise_type, data in noise_datasets.items():
        if len(data['syndromes']) > 0:
            test_syndromes = data['syndromes']
            test_errors = data['errors']
            
            # Evaluate
            decoder.eval()
            with torch.no_grad():
                predictions = decoder(test_syndromes)
                binary_preds = (predictions > 0.5).float()
                accuracy = (binary_preds == test_errors).float().mean().item()
            
            results[noise_type] = {
                'accuracy': accuracy,
                'n_samples': len(test_syndromes),
                'error_rate': test_errors.float().mean().item()
            }
    
    return results

def run_generalization_test(config_path: str = 'config/phase4_config.yaml'):
    """Run generalization test."""
    print("🔄 Running Generalization Test...")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load datasets
    noise_datasets = load_datasets_by_noise()
    
    # Load synthetic data
    with open('data/processed/synthetic_errors_fixed.pkl', 'rb') as f:
        synth_data = pickle.load(f)
    
    print(f"Loaded datasets for noise types: {list(noise_datasets.keys())}")
    
    # Train baseline (100% real depolarizing)
    print("\n1. Training baseline (100% real depolarizing)...")
    depol_syndromes = noise_datasets['depolarizing']['syndromes']
    depol_errors = noise_datasets['depolarizing']['errors']
    
    # Use 100% real for baseline
    baseline_decoder = QECDecoder(
        input_dim=depol_syndromes.shape[1],
        hidden_dim=64,
        output_dim=depol_errors.shape[1]
    )
    
    baseline_dataset = TensorDataset(depol_syndromes, depol_errors)
    baseline_loader = DataLoader(baseline_dataset, batch_size=64, shuffle=True)
    
    baseline_optimizer = optim.Adam(baseline_decoder.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    baseline_decoder.train()
    for epoch in range(100):
        for batch_s, batch_e in baseline_loader:
            baseline_optimizer.zero_grad()
            preds = baseline_decoder(batch_s)
            loss = criterion(preds, batch_e)
            loss.backward()
            baseline_optimizer.step()
        
        if epoch > 10 and loss.item() < 0.01:
            break
    
    # Train physics-informed decoder (30% real + 70% synthetic)
    print("2. Training physics-informed decoder (30% real + 70% synthetic)...")
    physics_decoder, n_real, n_synth = train_on_depolarizing(
        noise_datasets, synth_data, config
    )
    
    print(f"   Used {n_real} real samples and {n_synth} synthetic samples")
    
    # Test both decoders on all noise types
    print("\n3. Testing on different noise types...")
    
    baseline_results = test_on_noise_types(baseline_decoder, noise_datasets)
    physics_results = test_on_noise_types(physics_decoder, noise_datasets)
    
    # Calculate degradation
    results = []
    for noise_type in noise_datasets.keys():
        if noise_type in baseline_results and noise_type in physics_results:
            baseline_acc = baseline_results[noise_type]['accuracy']
            physics_acc = physics_results[noise_type]['accuracy']
            
            # Calculate degradation (negative means improvement)
            degradation = physics_acc - baseline_acc
            
            results.append({
                'noise_type': noise_type,
                'baseline_accuracy': baseline_acc,
                'physics_informed_accuracy': physics_acc,
                'accuracy_difference': degradation,
                'n_samples': baseline_results[noise_type]['n_samples'],
                'error_rate': baseline_results[noise_type]['error_rate']
            })
            
            status = "✅ IMPROVEMENT" if degradation > 0 else "⚠️  DEGRADATION"
            print(f"   {noise_type:20s} Baseline: {baseline_acc:.3f}, "
                  f"Physics-informed: {physics_acc:.3f}, "
                  f"Δ: {degradation:+.3f} {status}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    output_dir = Path("experiments/thrust3/generalization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "generalization_results.csv"
    df.to_csv(csv_path, index=False)
    
    json_path = output_dir / "generalization_results.json"
    df_dict = df.to_dict('records')
    with open(json_path, 'w') as f:
        json.dump(df_dict, f, indent=2)
    
    print(f"\n✅ Generalization test completed!")
    print(f"   Results saved to: {csv_path}")
    
    # Generate summary and check paper targets
    generate_generalization_summary(df, config)
    
    return df

def generate_generalization_summary(df, config):
    """Generate generalization test summary."""
    print("\n📊 GENERALIZATION TEST SUMMARY:")
    print("=" * 60)
    
    # Calculate average degradation for unseen noise types
    unseen_noise = ['amplitude_damping', 'phase_damping', 'combined']
    unseen_results = df[df['noise_type'].isin(unseen_noise)]
    
    if len(unseen_results) > 0:
        avg_degradation = unseen_results['accuracy_difference'].mean()
        max_degradation = unseen_results['accuracy_difference'].min()
        
        print(f"Unseen noise types: {unseen_noise}")
        print(f"Average accuracy difference: {avg_degradation:+.3f}")
        print(f"Maximum degradation: {max_degradation:+.3f}")
        
        # Check paper target
        paper_target = config['paper_targets']['generalization_degradation']
        
        print(f"\n🎯 PAPER TARGET: <{paper_target} degradation on unseen noise")
        
        if max_degradation > -paper_target:  # Less negative than target
            print(f"✅ SUCCESS: Maximum degradation ({max_degradation:.3f}) "
                  f"better than target (-{paper_target})")
        else:
            print(f"❌ FAILED: Maximum degradation ({max_degradation:.3f}) "
                  f"worse than target (-{paper_target})")
        
        print(f"\n📈 DETAILED RESULTS:")
        for _, row in df.iterrows():
            arrow = "↑" if row['accuracy_difference'] > 0 else "↓"
            print(f"  {row['noise_type']:20s} "
                  f"Baseline: {row['baseline_accuracy']:.3f}, "
                  f"Physics: {row['physics_informed_accuracy']:.3f} "
                  f"({arrow}{abs(row['accuracy_difference']):.3f})")
    
    print("=" * 60)

if __name__ == "__main__":
    results_df = run_generalization_test()
    
    # Create final assessment
    assessment = {
        'test_completed': True,
        'n_noise_types_tested': len(results_df),
        'paper_target_degradation': 0.10,
        'baseline_trained_on': '100% real depolarizing',
        'physics_informed_trained_on': '30% real + 70% synthetic depolarizing',
        'results_file': 'experiments/thrust3/generalization/generalization_results.csv'
    }
    
    # Calculate if target met
    unseen_noise = ['amplitude_damping', 'phase_damping', 'combined']
    unseen_results = results_df[results_df['noise_type'].isin(unseen_noise)]
    
    if len(unseen_results) > 0:
        max_degradation = abs(unseen_results['accuracy_difference'].min())
        assessment['max_degradation'] = float(max_degradation)
        assessment['target_met'] = max_degradation < 0.10
    else:
        assessment['target_met'] = False
    
    # Save assessment
    assessment_path = Path("experiments/thrust3/generalization/test_assessment.json")
    with open(assessment_path, 'w') as f:
        json.dump(assessment, f, indent=2, default=str)
    
    print(f"\n📋 Test assessment saved to: {assessment_path}")
    
    if assessment.get('target_met', False):
        print("🎉 GENERALIZATION TEST: SUCCESSFUL (Paper targets met!)")
    else:
        print("⚠️  GENERALIZATION TEST: Targets not fully met")
