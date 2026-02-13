"""
Final trainer for GNN with properly shaped data
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.thrust1.gnn_initializer.model.gnn_architecture_fixed import FixedParameterInitializer

class FinalGNNTrainingDataset(Dataset):
    """Dataset for GNN training with verified shapes"""
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        self.samples = self.data['samples']
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        n_qubits = sample['n_qubits']
        
        # Convert numpy arrays to torch tensors
        node_features = torch.FloatTensor(sample['node_features'])  # [n_qubits, 5]
        
        # Convert edge list to edge index format [2, num_edges]
        edge_list = sample['edge_list']
        if len(edge_list) > 0:
            edge_index = torch.LongTensor(edge_list).t().contiguous()
        else:
            # Create self-loops
            edge_index = torch.LongTensor([[i, i] for i in range(n_qubits)]).t().contiguous()
        
        # Targets: optimal parameters reshaped to [n_qubits, 3]
        optimal_params_flat = sample['optimal_params']  # Should be [n_qubits * 3]
        targets = torch.FloatTensor(optimal_params_flat).view(n_qubits, 3)  # [n_qubits, 3]
        
        # Gradient stats
        gradient_stats = sample['gradient_stats']
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'targets': targets,
            'n_qubits': n_qubits,
            'gradient_norm': gradient_stats['gradient_norm'],
            'trainable': gradient_stats['trainable'],
            'improvement_factor': gradient_stats['improvement_factor']
        }

def collate_fn(batch):
    """Custom collate function - process each graph individually"""
    return batch

def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        losses = []
        
        for sample in batch:
            # Move data to device
            node_features = sample['node_features'].to(device)
            edge_index = sample['edge_index'].to(device)
            targets = sample['targets'].to(device)
            
            # Forward pass
            predictions = model(node_features, edge_index)  # [n_qubits, 3]
            
            # Create gradient stats
            gradient_stats = {
                'gradient_norm': sample['gradient_norm'],
                'trainable': sample['trainable']
            }
            
            # Calculate loss
            loss = model.loss_function(predictions, targets, gradient_stats)
            losses.append(loss)
        
        # Average loss over batch
        if losses:
            batch_loss = torch.stack(losses).mean()
            
            # Backward pass
            optimizer.zero_grad()
            batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += batch_loss.item() * len(batch)
            total_samples += len(batch)
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    return avg_loss

def validate(model, dataloader, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_samples = 0
    total_improvement = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            for sample in batch:
                node_features = sample['node_features'].to(device)
                edge_index = sample['edge_index'].to(device)
                targets = sample['targets'].to(device)
                
                predictions = model(node_features, edge_index)
                
                gradient_stats = {
                    'gradient_norm': sample['gradient_norm'],
                    'trainable': sample['trainable']
                }
                
                loss = model.loss_function(predictions, targets, gradient_stats)
                total_loss += loss.item()
                total_improvement += sample['improvement_factor']
                total_samples += 1
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_improvement = total_improvement / total_samples if total_samples > 0 else 0
    
    return avg_loss, avg_improvement

def train_final_gnn(data_file, model_save_path, log_path, epochs=100, batch_size=32, 
                   learning_rate=0.001, checkpoint_dir=None):
    """Main training function"""
    print("Starting final GNN training...")
    print(f"Data file: {data_file}")
    print(f"Model save path: {model_save_path}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    
    # Create directories
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = FinalGNNTrainingDataset(data_file)
    print(f"Dataset size: {len(dataset)} samples")
    
    # Create train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 collate_fn=collate_fn, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               collate_fn=collate_fn, num_workers=0)
    
    # Create model
    model = FixedParameterInitializer(gnn_hidden_dim=64).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    train_losses = []
    val_losses = []
    val_improvements = []
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_improvement = validate(model, val_dataloader, device)
        val_losses.append(val_loss)
        val_improvements.append(val_improvement)
        
        # Update learning rate
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        print(f"Val Improvement Factor: {val_improvement:.1f}x")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        if checkpoint_dir and (epoch + 1) % 10 == 0:
            checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pt"
            model.save_model(checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_model(model_save_path)
            print(f"  Saved best model (val loss: {val_loss:.6f}, improvement: {val_improvement:.1f}x)")
    
    # Save training logs
    log_data = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_improvements': val_improvements,
        'best_val_loss': best_val_loss,
        'best_val_improvement': max(val_improvements) if val_improvements else 0,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'device': str(device)
    }
    
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss curves
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Improvement factor
    axes[1].plot(val_improvements, label='Val Improvement', linewidth=2, color='green')
    axes[1].axhline(y=1.0, color='red', linestyle='--', label='Baseline (1x)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Improvement Factor')
    axes[1].set_title('Improvement vs Random Initialization')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Loss (log scale)
    axes[2].semilogy(train_losses, label='Train Loss', linewidth=2)
    axes[2].semilogy(val_losses, label='Val Loss', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss (log scale)')
    axes[2].set_title('Loss (Log Scale)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = Path(log_path).parent / "training_curves_final.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best improvement factor: {max(val_improvements):.1f}x")
    print(f"Model saved to: {model_save_path}")
    print(f"Logs saved to: {log_path}")
    print(f"Training curves saved to: {plot_path}")
    
    return model, log_data

def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train final GNN for parameter initialization")
    parser.add_argument("--data_path", type=str, 
                       default="data/processed/gnn_training_dataset.pkl",
                       help="Path to training data")
    parser.add_argument("--model_save_path", type=str,
                       default="models/saved/gnn_initializer.pt",
                       help="Path to save trained model")
    parser.add_argument("--log_path", type=str,
                       default="logs/thrust1/training/gnn_training_final.json",
                       help="Path to save training logs")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str,
                       default="models/checkpoints/gnn/",
                       help="Directory for checkpoints")
    
    args = parser.parse_args()
    
    # Train the model
    train_final_gnn(
        data_file=args.data_path,
        model_save_path=args.model_save_path,
        log_path=args.log_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir
    )

if __name__ == "__main__":
    main()
