"""
Variational Autoencoder (VAE) Model
VAE for unsupervised representation learning of music features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VAE(nn.Module):
    """
    Variational Autoencoder for music feature representation learning.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128],
        latent_dim: int = 64,
        dropout: float = 0.2
    ):
        """
        Initialize VAE model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            latent_dim: Dimension of latent space
            dropout: Dropout rate
        """
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout = dropout
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space projections
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space parameters.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Tuple of (mu, logvar)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from latent space.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstruction.
        
        Args:
            z: Latent vector (batch_size, latent_dim)
            
        Returns:
            Reconstructed output (batch_size, input_dim)
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation (using mean).
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Latent vectors (batch_size, latent_dim)
        """
        mu, _ = self.encode(x)
        return mu


def vae_loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE loss function: reconstruction loss + KL divergence.
    
    Args:
        recon_x: Reconstructed output
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence (beta-VAE)
        
    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_loss)
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


class BetaVAE(VAE):
    """
    Beta-VAE variant with configurable beta parameter.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128],
        latent_dim: int = 64,
        dropout: float = 0.2,
        beta: float = 1.0
    ):
        """
        Initialize Beta-VAE.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            latent_dim: Dimension of latent space
            dropout: Dropout rate
            beta: Beta parameter for KL divergence weight
        """
        super(BetaVAE, self).__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout
        )
        self.beta = beta
    
    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute loss with configured beta.
        """
        return vae_loss_function(recon_x, x, mu, logvar, beta=self.beta)


def create_vae(
    input_dim: int,
    latent_dim: int = 64,
    hidden_dims: list = None,
    dropout: float = 0.2,
    beta: float = 1.0
) -> VAE:
    """
    Factory function to create VAE model.
    
    Args:
        input_dim: Dimension of input features
        latent_dim: Dimension of latent space
        hidden_dims: List of hidden layer dimensions (default: [256, 128])
        dropout: Dropout rate
        beta: Beta parameter for Beta-VAE
        
    Returns:
        VAE model
    """
    if hidden_dims is None:
        hidden_dims = [256, 128]
    
    if beta != 1.0:
        return BetaVAE(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout,
            beta=beta
        )
    else:
        return VAE(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout
        )


if __name__ == "__main__":
    # Test VAE model
    print("Testing VAE model...")
    
    # Create dummy data
    batch_size = 32
    input_dim = 500
    latent_dim = 64
    
    x = torch.randn(batch_size, input_dim)
    
    # Create model
    model = create_vae(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=[256, 128],
        dropout=0.2
    )
    
    print(f"\nModel architecture:")
    print(model)
    
    # Forward pass
    recon_x, mu, logvar = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Reconstruction shape: {recon_x.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    
    # Compute loss
    total_loss, recon_loss, kl_loss = vae_loss_function(recon_x, x, mu, logvar)
    
    print(f"\nLoss values:")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Reconstruction loss: {recon_loss.item():.4f}")
    print(f"  KL loss: {kl_loss.item():.4f}")
    
    # Get latent representation
    latent = model.get_latent(x)
    print(f"\nLatent representation shape: {latent.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {num_params:,}")
