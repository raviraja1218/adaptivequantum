"""
Train GNN for noise-aware parameter initialization
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

from src.thrust1.gnn_initializer.model.gnn_architecture import ParameterInitializer

class GNNTrainingDataset(Dataset):
    """Dataset for GNN training"""
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        self.samples = self.data['samples']
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert numpy arrays to torch tensors
        node_features = torch.FloatTensor(sample['node_features'])
        
        # Convert edge list to edge index format [2, num_edges]
        edge_list = sample['edge_list']
        if len(edge_list) > 0:
            edge_index = torch.LongTensor(edge_list).t().contiguous()
        else:
            # Create self-loops if no edges
            n_nodes = len(node_features)
            edge_index = torch.LongTensor([[i, i] for i in range(n_nodes)]).t().contiguous()
        
        # Targets: optimal parameters flattened
        targets = torch.FloatTensor(sample['optimal_params'].flatten())
        
        # Gradient stats for loss calculation
        gradient_stats = sample['gradient_stats']
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'targets': targets,
            'n_qubits': sample['n_qubits'],
            'gradient_norm': gradient_stats['gradient_norm'],
            'trainable': gradient_stats['trainable']
        }

def collate_fn(batch):
    """Custom collate function for variable-sized graphs"""
    # Since graphs have different sizes, we need to process them individually
    return batch

def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        # Process each graph individually (since sizes vary)
        losses = []
        
        for sample in batch:
            # Move data to device
            node_features = sample['node_features'].to(device)
            edge_index = sample['edge_index'].to(device)
            targets = sample['targets'].to(device)
            
            # Forward pass
            predictions = model(node_features, edge_index)
            
            # Reshape predictions to match targets
            predictions_flat = predictions.view(-1)
            
            # Create gradient stats dictionary
            gradient_stats = {
                'gradient_norm': sample['gradient_norm'],
                'trainable': sample['trainable']
            }
            
            # Calculate loss
            loss = model.loss_function(predictions_flat, targets, gradient_stats)
            losses.append(loss)
        
        # Average loss over batch
        if losses:
            batch_loss = torch.stack(losses).mean()
            
            # Backward pass
            optimizer.zero_grad()
            batch_loss.backward()
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
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            for sample in batch:
                node_features = sample['node_features'].to(device)
                edge_index = sample['edge_index'].to(device)
                targets = sample['targets'].to(device)
                
                predictions = model(node_features, edge_index)
                predictions_flat = predictions.view(-1)
                
                gradient_stats = {
                    'gradient_norm': sample['gradient_norm'],
                    'trainable': sample['trainable']
                }
                
                loss = model.loss_function(predictions_flat, targets, gradient_stats)
                total_loss += loss.item()
                total_samples += 1
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    return avg_loss

def train_gnn(data_file, model_save_path, log_path, epochs=100, batch_size=32, 
              learning_rate=0.001, checkpoint_dir=None):
    """Main training function"""
    print("Starting GNN training...")
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
    dataset = GNNTrainingDataset(data_file)
    print(f"Dataset size: {len(dataset)} samples")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           collate_fn=collate_fn, num_workers=0)
    
    # Create model
    model = ParameterInitializer(gnn_hidden_dim=64).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                    patience=10, factor=0.5)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss = train_epoch(model, dataloader, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate (use same data for now, in practice would have separate val set)
        val_loss = validate(model, dataloader, device)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        if checkpoint_dir and (epoch + 1) % 10 == 0:
            checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pt"
            model.save_model(checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_model(model_save_path)
            print(f"  Saved best model (val loss: {val_loss:.6f})")
    
    # Save training logs
    log_data = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'dataset_size': len(dataset),
        'device': str(device)
    }
    
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    # Plot training curves
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    ax[0].plot(train_losses, label='Train Loss', linewidth=2)
    ax[0].plot(val_losses, label='Val Loss', linewidth=2)
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training and Validation Loss')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    
    # Learning rate
    ax[1].semilogy(train_losses, label='Train Loss', linewidth=2)
    ax[1].semilogy(val_losses, label='Val Loss', linewidth=2)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss (log scale)')
    ax[1].set_title('Loss (Log Scale)')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = Path(log_path).parent / "training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {model_save_path}")
    print(f"Logs saved to: {log_path}")
    print(f"Training curves saved to: {plot_path}")
    
    return model, log_data

def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train GNN for parameter initialization")
    parser.add_argument("--data_path", type=str, 
                       default="data/processed/gnn_training_dataset.pkl",
                       help="Path to training data")
    parser.add_argument("--model_save_path", type=str,
                       default="models/saved/gnn_initializer.pt",
                       help="Path to save trained model")
    parser.add_argument("--log_path", type=str,
                       default="logs/thrust1/training/gnn_training.json",
                       help="Path to save training logs")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str,
                       default="models/checkpoints/gnn/",
                       help="Directory for checkpoints")
    
    args = parser.parse_args()
    
    # Train the model
    train_gnn(
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
