"""
Conditional Variational Autoencoder for synthetic error generation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConditionalVAE(nn.Module):
    """Conditional VAE for generating quantum errors conditioned on noise parameters."""
    
    def __init__(self, syndrome_dim: int = 12, latent_dim: int = 16, 
                 noise_param_dim: int = 4, hidden_dims: list = [128, 64]):
        super().__init__()
        
        self.syndrome_dim = syndrome_dim
        self.latent_dim = latent_dim
        self.noise_param_dim = noise_param_dim
        
        # Encoder: q(z | syndrome, noise_params)
        encoder_layers = []
        input_dim = syndrome_dim + noise_param_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
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
            input_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(hidden_dims[0], syndrome_dim))
        decoder_layers.append(nn.Sigmoid())  # Output probabilities
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, syndrome: torch.Tensor, noise_params: torch.Tensor) -> tuple:
        """Encode syndrome and noise parameters to latent distribution."""
        x = torch.cat([syndrome, noise_params], dim=1)
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
    
    def forward(self, syndrome: torch.Tensor, noise_params: torch.Tensor, 
                errors: torch.Tensor = None):
        """Forward pass through the VAE."""
        mu, logvar = self.encode(syndrome, noise_params)
        z = self.reparameterize(mu, logvar)
        
        # Decode to error probabilities
        error_probs = self.decode(z, noise_params)
        
        if errors is not None:
            # Compute reconstruction loss (binary cross entropy)
            recon_loss = F.binary_cross_entropy(error_probs, errors, reduction='sum')
            
            # Compute KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            return error_probs, recon_loss, kl_loss, mu, logvar
        else:
            return error_probs
    
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
            
            # Decode to error probabilities
            error_probs = self.decode(z, noise_params)
            
            # Sample actual errors from Bernoulli distribution
            errors = torch.bernoulli(error_probs)
            
        return errors, error_probs
    
    def generate_from_syndrome(self, syndrome: torch.Tensor, 
                              noise_params: torch.Tensor) -> torch.Tensor:
        """Generate error distribution for a given syndrome."""
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(syndrome, noise_params)
            error_probs = self.decode(mu, noise_params)
        return error_probs

class NoiseParameterEncoder:
    """Encode noise types to parameter vectors."""
    
    def __init__(self):
        # Define noise parameter mapping
        self.noise_params = {
            'depolarizing': [0.01, 0.008, 0.001, 0.0005],      # T1, T2, depol, dephasing
            'amplitude_damping': [0.005, 0.004, 0.0001, 0.001],
            'phase_damping': [0.015, 0.003, 0.0005, 0.002],
            'combined': [0.01, 0.005, 0.0008, 0.0015]
        }
    
    def encode(self, noise_type: str) -> torch.Tensor:
        """Encode noise type to parameter vector."""
        if noise_type in self.noise_params:
            return torch.tensor(self.noise_params[noise_type], dtype=torch.float32)
        else:
            # Default to depolarizing
            return torch.tensor(self.noise_params['depolarizing'], dtype=torch.float32)
    
    def sample_random_params(self, n_samples: int = 1) -> torch.Tensor:
        """Sample random noise parameters within realistic ranges."""
        params = []
        for _ in range(n_samples):
            # Sample realistic hardware parameters
            t1 = torch.rand(1).item() * 0.02 + 0.001  # 1-20 ms
            t2 = torch.rand(1).item() * 0.015 + 0.001  # 1-15 ms
            depol = torch.rand(1).item() * 0.002 + 0.0001  # 0.01-0.2%
            dephase = torch.rand(1).item() * 0.003 + 0.0001  # 0.01-0.3%
            params.append([t1, t2, depol, dephase])
        
        return torch.tensor(params, dtype=torch.float32)

def test_vae():
    """Test the VAE implementation."""
    print("Testing Conditional VAE...")
    
    # Create model
    vae = ConditionalVAE(syndrome_dim=12, latent_dim=16, noise_param_dim=4)
    
    # Test inputs
    batch_size = 32
    syndrome = torch.randn(batch_size, 12)
    noise_params = torch.randn(batch_size, 4)
    errors = torch.randint(0, 2, (batch_size, 12)).float()
    
    # Forward pass
    error_probs, recon_loss, kl_loss, mu, logvar = vae(syndrome, noise_params, errors)
    
    print(f"Model architecture: {vae}")
    print(f"Input syndrome shape: {syndrome.shape}")
    print(f"Output error_probs shape: {error_probs.shape}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")
    
    # Test sampling
    noise_encoder = NoiseParameterEncoder()
    test_params = noise_encoder.encode('depolarizing').unsqueeze(0)
    samples, probs = vae.sample(5, test_params)
    
    print(f"\nGenerated samples shape: {samples.shape}")
    print(f"Sample probabilities shape: {probs.shape}")
    print("✅ VAE implementation test passed!")

if __name__ == "__main__":
    test_vae()
