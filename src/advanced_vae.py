"""
Advanced VAE Architectures
Includes Convolutional VAE, Conditional VAE, and Beta-VAE variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class ConvVAE(nn.Module):
    """
    Convolutional VAE for spectrogram/MFCC features.
    Assumes input is 2D image-like representation (e.g., mel-spectrogram).
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        input_height: int = 128,
        input_width: int = 128,
        latent_dim: int = 64,
        beta: float = 1.0
    ):
        """
        Initialize Convolutional VAE.
        
        Args:
            input_channels: Number of input channels (1 for mono)
            input_height: Height of input spectrogram
            input_width: Width of input spectrogram
            latent_dim: Dimension of latent space
            beta: Beta parameter for KL divergence weight
        """
        super(ConvVAE, self).__init__()
        
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder
        self.encoder = nn.Sequential(
            # Conv1: (1, 128, 128) -> (32, 64, 64)
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Conv2: (32, 64, 64) -> (64, 32, 32)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Conv3: (64, 32, 32) -> (128, 16, 16)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Conv4: (128, 16, 16) -> (256, 8, 8)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Calculate flattened size
        self.flattened_size = 256 * 8 * 8
        
        # Latent projections
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flattened_size)
        
        self.decoder = nn.Sequential(
            # Deconv1: (256, 8, 8) -> (128, 16, 16)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Deconv2: (128, 16, 16) -> (64, 32, 32)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Deconv3: (64, 32, 32) -> (32, 64, 64)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Deconv4: (32, 64, 64) -> (1, 128, 128)
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters."""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        h = self.decoder_input(z)
        h = h.view(h.size(0), 256, 8, 8)
        return self.decoder(h)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through ConvVAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation (using mean)."""
        mu, _ = self.encode(x)
        return mu


class ConditionalVAE(nn.Module):
    """
    Conditional VAE (CVAE) that conditions on genre/label information.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list = [256, 128],
        latent_dim: int = 64,
        dropout: float = 0.2
    ):
        """
        Initialize Conditional VAE.
        
        Args:
            input_dim: Dimension of input features
            num_classes: Number of condition classes (e.g., genres)
            hidden_dims: List of hidden layer dimensions
            latent_dim: Dimension of latent space
            dropout: Dropout rate
        """
        super(ConditionalVAE, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        
        # Encoder (input + condition)
        encoder_layers = []
        prev_dim = input_dim + num_classes
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent projections
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder (latent + condition)
        decoder_layers = []
        prev_dim = latent_dim + num_classes
        
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
    
    def encode(
        self,
        x: torch.Tensor,
        c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input conditioned on class label.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            c: Condition tensor (batch_size, num_classes) - one-hot encoded
            
        Returns:
            Tuple of (mu, logvar)
        """
        # Concatenate input with condition
        xc = torch.cat([x, c], dim=1)
        h = self.encoder(xc)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector conditioned on class label.
        
        Args:
            z: Latent vector (batch_size, latent_dim)
            c: Condition tensor (batch_size, num_classes)
            
        Returns:
            Reconstructed output
        """
        zc = torch.cat([z, c], dim=1)
        return self.decoder(zc)
    
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through CVAE.
        
        Args:
            x: Input tensor
            c: Condition tensor (one-hot encoded)
            
        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, c)
        return reconstruction, mu, logvar
    
    def get_latent(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Get latent representation."""
        mu, _ = self.encode(x, c)
        return mu


class DisentangledBetaVAE(nn.Module):
    """
    Disentangled Beta-VAE for learning interpretable latent representations.
    Uses higher beta values to encourage disentanglement.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128],
        latent_dim: int = 64,
        dropout: float = 0.2,
        beta: float = 4.0
    ):
        """
        Initialize Disentangled Beta-VAE.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            latent_dim: Dimension of latent space
            dropout: Dropout rate
            beta: Beta parameter (higher = more disentangled, typically 2-10)
        """
        super(DisentangledBetaVAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder
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
        
        # Latent projections
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
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
        """Encode input to latent parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation."""
        mu, _ = self.encode(x)
        return mu
    
    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Beta-VAE loss with higher weight on KL divergence.
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        # Total loss with beta weighting
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss


def create_advanced_vae(
    vae_type: str,
    input_dim: int,
    latent_dim: int = 64,
    num_classes: int = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create advanced VAE models.
    
    Args:
        vae_type: Type of VAE ('conv', 'conditional', 'disentangled')
        input_dim: Input dimension
        latent_dim: Latent dimension
        num_classes: Number of classes (for conditional VAE)
        **kwargs: Additional arguments
        
    Returns:
        VAE model
    """
    if vae_type == 'conv':
        # For convolutional VAE, input_dim should be (height, width)
        if isinstance(input_dim, tuple):
            height, width = input_dim
        else:
            # Assume square
            height = width = int(np.sqrt(input_dim))
        
        return ConvVAE(
            input_height=height,
            input_width=width,
            latent_dim=latent_dim,
            **kwargs
        )
    
    elif vae_type == 'conditional':
        if num_classes is None:
            raise ValueError("num_classes required for conditional VAE")
        
        return ConditionalVAE(
            input_dim=input_dim,
            num_classes=num_classes,
            latent_dim=latent_dim,
            **kwargs
        )
    
    elif vae_type == 'disentangled':
        return DisentangledBetaVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            beta=kwargs.get('beta', 4.0),
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown VAE type: {vae_type}")


if __name__ == "__main__":
    print("Testing advanced VAE architectures...")
    
    # Test Conditional VAE
    batch_size = 16
    input_dim = 500
    num_classes = 10
    latent_dim = 64
    
    print("\n1. Testing Conditional VAE...")
    cvae = ConditionalVAE(
        input_dim=input_dim,
        num_classes=num_classes,
        latent_dim=latent_dim
    )
    
    x = torch.randn(batch_size, input_dim)
    c = torch.zeros(batch_size, num_classes)
    c[range(batch_size), torch.randint(0, num_classes, (batch_size,))] = 1  # One-hot
    
    recon, mu, logvar = cvae(x, c)
    print(f"  Input: {x.shape}")
    print(f"  Condition: {c.shape}")
    print(f"  Reconstruction: {recon.shape}")
    print(f"  Latent: {mu.shape}")
    
    # Test Disentangled Beta-VAE
    print("\n2. Testing Disentangled Beta-VAE...")
    dbvae = DisentangledBetaVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        beta=4.0
    )
    
    recon, mu, logvar = dbvae(x)
    print(f"  Input: {x.shape}")
    print(f"  Reconstruction: {recon.shape}")
    print(f"  Latent: {mu.shape}")
    
    print("\nAll tests passed!")
