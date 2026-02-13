"""
Fixed Conditional VAE for synthetic error generation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConditionalVAEFixed(nn.Module):
    """Fixed Conditional VAE for generating quantum errors."""
    
    def __init__(self, error_dim: int = 9, latent_dim: int = 16, 
                 noise_param_dim: int = 4, hidden_dims: list = [128, 64]):
        super().__init__()
        
        self.error_dim = error_dim
        self.latent_dim = latent_dim
        self.noise_param_dim = noise_param_dim
        
        # Encoder: q(z | error, noise_params)
        encoder_layers = []
        input_dim = error_dim + noise_param_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Mean and log variance for latent distribution
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder: p(error | z, noise_params)
        decoder_layers = []
        input_dim = latent_dim + noise_param_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(input_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(hidden_dims[0], error_dim))
        decoder_layers.append(nn.Sigmoid())  # Output probabilities
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, error: torch.Tensor, noise_params: torch.Tensor) -> tuple:
        """Encode error and noise parameters to latent distribution."""
        x = torch.cat([error, noise_params], dim=1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, noise_params: torch.Tensor) -> torch.Tensor:
        """Decode latent vector and noise parameters to error probabilities."""
        x = torch.cat([z, noise_params], dim=1)
        return self.decoder(x)
    
    def forward(self, error: torch.Tensor, noise_params: torch.Tensor):
        """Forward pass through the VAE."""
        mu, logvar = self.encode(error, noise_params)
        z = self.reparameterize(mu, logvar)
        
        # Decode to error probabilities
        error_probs = self.decode(z, noise_params)
        
        # Compute reconstruction loss (binary cross entropy)
        recon_loss = F.binary_cross_entropy(error_probs, error, reduction='sum')
        
        # Compute KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return error_probs, recon_loss, kl_loss, mu, logvar
    
    def sample(self, n_samples: int, noise_params: torch.Tensor, 
               device: str = 'cpu') -> torch.Tensor:
        """Generate synthetic errors by sampling from latent space."""
        self.eval()
        with torch.no_grad():
            # Sample from standard normal
            z = torch.randn(n_samples, self.latent_dim).to(device)
            
            # Repeat noise_params for each sample if needed
            if noise_params.shape[0] == 1:
                noise_params = noise_params.repeat(n_samples, 1)
            else:
                noise_params = noise_params.to(device)
            
            # Decode to error probabilities
            error_probs = self.decode(z, noise_params)
            
            # Sample actual errors from Bernoulli distribution
            errors = torch.bernoulli(error_probs)
            
        return errors, error_probs

class NoiseParameterEncoderFixed:
    """Fixed noise parameter encoder."""
    
    def __init__(self):
        # Define noise parameter mapping (T1, T2, depol_rate, dephase_rate)
        self.noise_params = {
            'depolarizing': torch.tensor([0.010, 0.008, 0.0010, 0.0005], dtype=torch.float32),
            'amplitude_damping': torch.tensor([0.005, 0.004, 0.0001, 0.0010], dtype=torch.float32),
            'phase_damping': torch.tensor([0.015, 0.003, 0.0005, 0.0020], dtype=torch.float32),
            'combined': torch.tensor([0.010, 0.005, 0.0008, 0.0015], dtype=torch.float32)
        }
    
    def encode(self, noise_type: str) -> torch.Tensor:
        """Encode noise type to parameter vector."""
        return self.noise_params.get(noise_type, self.noise_params['depolarizing'])
    
    def encode_batch(self, noise_labels):
        """Encode a batch of noise labels."""
        params = []
        for label in noise_labels:
            params.append(self.encode(label).numpy())
        return torch.tensor(np.array(params), dtype=torch.float32)

def test_vae_fixed():
    """Test the fixed VAE implementation."""
    print("Testing Fixed Conditional VAE...")
    
    # Create model
    vae = ConditionalVAEFixed(error_dim=9, latent_dim=16, noise_param_dim=4)
    
    # Test inputs
    batch_size = 32
    errors = torch.randint(0, 2, (batch_size, 9)).float()
    noise_params = torch.randn(batch_size, 4)
    
    # Forward pass
    error_probs, recon_loss, kl_loss, mu, logvar = vae(errors, noise_params)
    
    print(f"Model architecture: {vae}")
    print(f"Input errors shape: {errors.shape}")
    print(f"Output error_probs shape: {error_probs.shape}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")
    
    # Test sampling
    noise_encoder = NoiseParameterEncoderFixed()
    test_params = noise_encoder.encode('depolarizing').unsqueeze(0)
    samples, probs = vae.sample(5, test_params)
    
    print(f"\nGenerated samples shape: {samples.shape}")
    print(f"Sample probabilities shape: {probs.shape}")
    print("✅ Fixed VAE implementation test passed!")

if __name__ == "__main__":
    test_vae_fixed()
