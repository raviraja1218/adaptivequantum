"""
Train discriminator to validate synthetic data quality.
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

sys.path.append(str(Path(__file__).parent.parent.parent))

class Discriminator(nn.Module):
    """Discriminator network to distinguish real vs synthetic errors."""
    
    def __init__(self, input_dim: int = 9, hidden_dim: int = 64):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def load_training_data():
    """Load real and synthetic data for discriminator training."""
    # Load real data
    with open('data/processed/qec_dataset.pkl', 'rb') as f:
        real_data = pickle.load(f)
    
    # Load synthetic data
    with open('data/processed/synthetic_errors_fixed.pkl', 'rb') as f:
        synth_data = pickle.load(f)
    
    # Use training split of real data
    real_errors = real_data['train']['errors']
    real_labels = torch.ones(len(real_errors), 1)
    
    # Use subset of synthetic data
    synth_errors = synth_data['errors'][:len(real_errors)]  # Balance datasets
    synth_labels = torch.zeros(len(synth_errors), 1)
    
    # Combine
    all_errors = torch.cat([real_errors, synth_errors], dim=0)
    all_labels = torch.cat([real_labels, synth_labels], dim=0)
    
    # Shuffle
    indices = torch.randperm(len(all_errors))
    all_errors = all_errors[indices]
    all_labels = all_labels[indices]
    
    return all_errors, all_labels, real_errors, synth_errors

def train_discriminator(config_path: str = 'config/phase4_config.yaml'):
    """Train discriminator to distinguish real vs synthetic."""
    print("🔄 Training Discriminator...")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    train_errors, train_labels, real_errors, synth_errors = load_training_data()
    
    print(f"Training samples: {len(train_errors)}")
    print(f"  Real: {len(real_errors)}")
    print(f"  Synthetic: {len(synth_errors)}")
    
    # Create dataset and dataloader
    dataset = TensorDataset(train_errors, train_labels)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Initialize discriminator
    discriminator = Discriminator(input_dim=train_errors.shape[1], hidden_dim=64)
    
    # Setup training
    optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # Training loop
    discriminator.train()
    train_losses = []
    train_accuracies = []
    
    n_epochs = 50
    for epoch in range(n_epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_errors, batch_labels in dataloader:
            optimizer.zero_grad()
            
            predictions = discriminator(batch_errors)
            loss = criterion(predictions, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Calculate accuracy
            binary_preds = (predictions > 0.5).float()
            correct += (binary_preds == batch_labels).sum().item()
            total += len(batch_labels)
        
        # Calculate metrics
        avg_loss = epoch_loss / len(dataloader)
        accuracy = correct / total
        
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: Loss = {avg_loss:.4f}, "
                  f"Accuracy = {accuracy:.3f}")
    
    print(f"\n✅ Discriminator training completed!")
    print(f"   Final accuracy: {train_accuracies[-1]:.3f}")
    
    # Test on validation data
    test_discriminator(discriminator, real_errors, synth_errors)
    
    # Save results
    save_discriminator_results(discriminator, train_losses, train_accuracies, config)
    
    return discriminator, train_accuracies[-1]

def test_discriminator(discriminator, real_errors, synth_errors):
    """Test discriminator on held-out data."""
    print("\n🧪 Testing Discriminator...")
    
    # Load validation data
    with open('data/processed/qec_dataset.pkl', 'rb') as f:
        real_data = pickle.load(f)
    
    with open('data/processed/synthetic_errors_fixed.pkl', 'rb') as f:
        synth_data = pickle.load(f)
    
    # Use validation split
    val_real_errors = real_data['val']['errors']
    val_synth_errors = synth_data['errors'][len(real_data['train']['errors']): 
                                           len(real_data['train']['errors']) + len(val_real_errors)]
    
    # Create test dataset
    test_errors = torch.cat([val_real_errors, val_synth_errors], dim=0)
    test_labels = torch.cat([torch.ones(len(val_real_errors), 1), 
                            torch.zeros(len(val_synth_errors), 1)], dim=0)
    
    # Evaluate
    discriminator.eval()
    with torch.no_grad():
        predictions = discriminator(test_errors)
        binary_preds = (predictions > 0.5).float()
        
        accuracy = (binary_preds == test_labels).float().mean().item()
        
        # Calculate confusion matrix
        tp = ((binary_preds == 1) & (test_labels == 1)).sum().item()
        fp = ((binary_preds == 1) & (test_labels == 0)).sum().item()
        tn = ((binary_preds == 0) & (test_labels == 0)).sum().item()
        fn = ((binary_preds == 0) & (test_labels == 1)).sum().item()
    
    print(f"  Test accuracy: {accuracy:.3f}")
    print(f"  Confusion matrix:")
    print(f"    True Positives: {tp}")
    print(f"    False Positives: {fp}")
    print(f"    True Negatives: {tn}")
    print(f"    False Negatives: {fn}")
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1-Score: {f1:.3f}")
    
    # Determine if synthetic data is indistinguishable
    print(f"\n🎯 SYNTHETIC DATA ASSESSMENT:")
    if accuracy < 0.55:  # Close to random chance
        print(f"  ✅ Discriminator accuracy: {accuracy:.3f} < 0.55")
        print(f"  ✅ Synthetic data is INDISTINGUISHABLE from real data!")
        indistinguishable = True
    else:
        print(f"  ⚠️  Discriminator accuracy: {accuracy:.3f} ≥ 0.55")
        print(f"  ⚠️  Synthetic data is somewhat distinguishable")
        indistinguishable = False
    
    return accuracy, indistinguishable, {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

def save_discriminator_results(discriminator, losses, accuracies, config):
    """Save discriminator results."""
    output_dir = Path("experiments/thrust3/discriminator")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test discriminator
    _, _, real_errors, synth_errors = load_training_data()
    test_acc, indistinguishable, confusion = test_discriminator(discriminator, real_errors, synth_errors)
    
    # Create results dictionary
    results = {
        'training_loss': [float(l) for l in losses],
        'training_accuracy': [float(a) for a in accuracies],
        'final_training_accuracy': float(accuracies[-1]),
        'test_accuracy': float(test_acc),
        'indistinguishable': bool(indistinguishable),
        'confusion_matrix': confusion,
        'paper_target': 'Discriminator accuracy < 0.55',
        'target_met': test_acc < 0.55,
        'synthetic_data_quality': 'excellent' if test_acc < 0.55 else 'good' if test_acc < 0.65 else 'needs_improvement'
    }
    
    # Save results
    results_path = output_dir / "discriminator_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Discriminator results saved to: {results_path}")
    
    return results

if __name__ == "__main__":
    discriminator, final_acc = train_discriminator()
    
    # Final assessment
    print("\n📋 FINAL DISCRIMINATOR ASSESSMENT:")
    print("=" * 50)
    
    results_path = Path("experiments/thrust3/discriminator/discriminator_results.json")
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        if results['target_met']:
            print("✅ PAPER TARGET MET: Discriminator accuracy < 0.55")
            print(f"   Actual accuracy: {results['test_accuracy']:.3f}")
            print("   Synthetic data quality: EXCELLENT")
        else:
            print("⚠️  PAPER TARGET NOT MET: Discriminator accuracy ≥ 0.55")
            print(f"   Actual accuracy: {results['test_accuracy']:.3f}")
            print(f"   Synthetic data quality: {results['synthetic_data_quality'].upper()}")
    
    print("=" * 50)
