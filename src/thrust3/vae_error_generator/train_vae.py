"""
Train conditional VAE for synthetic error generation.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from pathlib import Path
import sys
import yaml
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.thrust3.vae_error_generator.model.conditional_vae import ConditionalVAE, NoiseParameterEncoder

def load_dataset():
    """Load the prepared QEC dataset."""
    with open('data/processed/qec_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    
    return dataset

def create_noise_params(noise_labels):
    """Create noise parameter vectors from noise labels."""
    encoder = NoiseParameterEncoder()
    params = []
    
    for label in noise_labels:
        param = encoder.encode(label)
        params.append(param.numpy())
    
    return torch.tensor(params, dtype=torch.float32)

def train_vae(config_path: str = 'config/phase4_config.yaml'):
    """Train conditional VAE for synthetic error generation."""
    print("🔄 Training Conditional VAE...")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    vae_config = config['vae']
    
    # Load dataset
    dataset = load_dataset()
    
    # Prepare training data
    train_syndromes = dataset['train']['syndromes']
    train_errors = dataset['train']['errors']
    train_noise_labels = dataset['train']['noise_labels']
    
    # Create noise parameters
    train_noise_params = create_noise_params(train_noise_labels)
    
    # Create validation data
    val_syndromes = dataset['val']['syndromes']
    val_errors = dataset['val']['errors']
    val_noise_labels = dataset['val']['noise_labels']
    val_noise_params = create_noise_params(val_noise_labels)
    
    # Create data loaders
    train_dataset = TensorDataset(train_syndromes, train_noise_params, train_errors)
    val_dataset = TensorDataset(val_syndromes, val_noise_params, val_errors)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=vae_config['batch_size'], 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=vae_config['batch_size'], 
        shuffle=False
    )
    
    # Initialize model
    syndrome_dim = train_syndromes.shape[1]
    vae = ConditionalVAE(
        syndrome_dim=syndrome_dim,
        latent_dim=vae_config['latent_dim'],
        noise_param_dim=4,
        hidden_dims=vae_config['hidden_dims']
    )
    
    # Setup optimizer
    optimizer = optim.Adam(vae.parameters(), lr=vae_config['learning_rate'])
    
    # Training loop with β scheduling
    epochs = vae_config['epochs']
    beta_start = vae_config['beta_start']
    beta_end = vae_config['beta_end']
    
    train_losses = []
    val_losses = []
    recon_losses = []
    kl_losses = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = vae.to(device)
    
    print(f"Training on: {device}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    for epoch in range(epochs):
        # β scheduling (linear increase)
        beta = beta_start + (beta_end - beta_start) * (epoch / epochs)
        
        # Training
        vae.train()
        train_loss = 0
        train_recon = 0
        train_kl = 0
        
        for syndromes, noise_params, errors in tqdm(train_loader, 
                                                    desc=f'Epoch {epoch+1}/{epochs}'):
            syndromes = syndromes.to(device)
            noise_params = noise_params.to(device)
            errors = errors.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            _, recon_loss, kl_loss, _, _ = vae(syndromes, noise_params, errors)
            
            # β-VAE loss
            loss = recon_loss + beta * kl_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()
        
        # Validation
        vae.eval()
        val_loss = 0
        val_recon = 0
        val_kl = 0
        
        with torch.no_grad():
            for syndromes, noise_params, errors in val_loader:
                syndromes = syndromes.to(device)
                noise_params = noise_params.to(device)
                errors = errors.to(device)
                
                _, recon_loss, kl_loss, _, _ = vae(syndromes, noise_params, errors)
                loss = recon_loss + beta * kl_loss
                
                val_loss += loss.item()
                val_recon += recon_loss.item()
                val_kl += kl_loss.item()
        
        # Average losses
        train_loss /= len(train_loader)
        train_recon /= len(train_loader)
        train_kl /= len(train_loader)
        
        val_loss /= len(val_loader)
        val_recon /= len(val_loader)
        val_kl /= len(val_loader)
        
        # Store losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        recon_losses.append(train_recon)
        kl_losses.append(train_kl)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}/{epochs}:")
            print(f"  β: {beta:.3f}")
            print(f"  Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f})")
            print(f"  Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})")
        
        # Save checkpoint
        if (epoch + 1) % 50 == 0:
            checkpoint_dir = Path("models/checkpoints/conditional_vae")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'beta': beta
            }, checkpoint_path)
            
            print(f"  Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    models_dir = Path("models/saved")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / "conditional_vae.pt"
    torch.save(vae.state_dict(), model_path)
    
    print(f"\n✅ Training completed!")
    print(f"   Final model saved to: {model_path}")
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'recon_loss': recon_losses,
        'kl_loss': kl_losses,
        'beta_schedule': [beta_start + (beta_end - beta_start) * (i/epochs) 
                         for i in range(epochs)],
        'config': vae_config
    }
    
    history_path = Path("logs/thrust3/training/vae_training.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"   Training history saved to: {history_path}")
    
    # Generate training curves
    generate_training_curves(history, history_path.parent)
    
    return vae, history

def generate_training_curves(history, output_dir):
    """Generate and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Total loss
    axes[0, 0].plot(epochs, history['train_loss'], label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Reconstruction loss
    axes[0, 1].plot(epochs, history['recon_loss'])
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reconstruction Loss')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: KL divergence
    axes[1, 0].plot(epochs, history['kl_loss'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].set_title('KL Divergence')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: β schedule
    axes[1, 1].plot(epochs, history['beta_schedule'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('β Value')
    axes[1, 1].set_title('β Schedule')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = output_dir / "vae_training_curves.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Training curves saved to: {fig_path}")

if __name__ == "__main__":
    vae, history = train_vae()
