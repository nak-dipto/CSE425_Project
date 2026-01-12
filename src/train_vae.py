"""
VAE Training Module
Train Variational Autoencoder on fused music features.
"""

import os
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from vae import VAE, vae_loss_function


class VAETrainer:
    """
    Trainer for VAE model.
    """
    
    def __init__(
        self,
        model: VAE,
        device: str = None,
        learning_rate: float = 1e-3,
        beta: float = 1.0
    ):
        """
        Initialize VAE trainer.
        
        Args:
            model: VAE model
            device: Device to train on (cuda/cpu)
            learning_rate: Learning rate for optimizer
            beta: Beta parameter for KL divergence weight
        """
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.beta = beta
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        self.history = {
            "train_loss": [],
            "train_recon_loss": [],
            "train_kl_loss": [],
            "val_loss": [],
            "val_recon_loss": [],
            "val_kl_loss": []
        }
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        verbose: bool = True
    ) -> Tuple[float, float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            verbose: Show progress bar
            
        Returns:
            Tuple of (avg_loss, avg_recon_loss, avg_kl_loss)
        """
        self.model.train()
        
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        num_batches = 0
        
        iterator = tqdm(train_loader, desc="Training") if verbose else train_loader
        
        for batch in iterator:
            x = batch[0].to(self.device)
            
            # Forward pass
            recon_x, mu, logvar = self.model(x)
            
            # Compute loss
            loss, recon_loss, kl_loss = vae_loss_function(
                recon_x, x, mu, logvar, beta=self.beta
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1
            
            if verbose:
                iterator.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'kl': f'{kl_loss.item():.4f}'
                })
        
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        
        return avg_loss, avg_recon_loss, avg_kl_loss
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, float, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (avg_loss, avg_recon_loss, avg_kl_loss)
        """
        self.model.eval()
        
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(self.device)
                
                # Forward pass
                recon_x, mu, logvar = self.model(x)
                
                # Compute loss
                loss, recon_loss, kl_loss = vae_loss_function(
                    recon_x, x, mu, logvar, beta=self.beta
                )
                
                # Accumulate losses
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        
        return avg_loss, avg_recon_loss, avg_kl_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 25,
        verbose: bool = True
    ):
        """
        Train VAE model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs
            verbose: Show progress
        """
        print(f"Training VAE on {self.device}")
        print(f"Epochs: {epochs}, Learning rate: {self.learning_rate}, Beta: {self.beta}")
        
        for epoch in range(epochs):
            # Train
            train_loss, train_recon, train_kl = self.train_epoch(
                train_loader, verbose=verbose
            )
            
            self.history["train_loss"].append(train_loss)
            self.history["train_recon_loss"].append(train_recon)
            self.history["train_kl_loss"].append(train_kl)
            
            # Validate
            if val_loader is not None:
                val_loss, val_recon, val_kl = self.validate(val_loader)
                self.history["val_loss"].append(val_loss)
                self.history["val_recon_loss"].append(val_recon)
                self.history["val_kl_loss"].append(val_kl)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f}) - "
                          f"Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f})")
    
    def get_latent_representations(
        self,
        data_loader: DataLoader
    ) -> np.ndarray:
        """
        Extract latent representations for entire dataset.
        
        Args:
            data_loader: Data loader
            
        Returns:
            Latent representations (n_samples, latent_dim)
        """
        self.model.eval()
        latents = []
        
        with torch.no_grad():
            for batch in data_loader:
                x = batch[0].to(self.device)
                mu, _ = self.model.encode(x)
                latents.append(mu.cpu().numpy())
        
        return np.vstack(latents)
    
    def save_model(self, path: str):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load model checkpoint.
        
        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Model loaded from {path}")
    
    def plot_training_history(self, save_path: str = None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save plot (optional)
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Total loss
        axes[0].plot(self.history["train_loss"], label="Train")
        if self.history["val_loss"]:
            axes[0].plot(self.history["val_loss"], label="Val")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Total Loss")
        axes[0].set_title("Total Loss")
        axes[0].legend()
        axes[0].grid(True)
        
        # Reconstruction loss
        axes[1].plot(self.history["train_recon_loss"], label="Train")
        if self.history["val_recon_loss"]:
            axes[1].plot(self.history["val_recon_loss"], label="Val")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Reconstruction Loss")
        axes[1].set_title("Reconstruction Loss")
        axes[1].legend()
        axes[1].grid(True)
        
        # KL loss
        axes[2].plot(self.history["train_kl_loss"], label="Train")
        if self.history["val_kl_loss"]:
            axes[2].plot(self.history["val_kl_loss"], label="Val")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("KL Divergence Loss")
        axes[2].set_title("KL Divergence Loss")
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.close()


def prepare_data_loaders(
    features: np.ndarray,
    batch_size: int = 32,
    val_split: float = 0.2,
    shuffle: bool = True,
    random_state: int = 42
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Prepare data loaders for training.
    
    Args:
        features: Feature matrix
        batch_size: Batch size
        val_split: Validation split ratio
        shuffle: Whether to shuffle training data
        random_state: Random state for shuffling
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Convert to torch tensors
    features_tensor = torch.FloatTensor(features)
    
    # Split into train and validation
    if val_split > 0:
        n_samples = len(features)
        n_val = int(n_samples * val_split)
        
        # Shuffle and split
        np.random.seed(random_state)
        indices = np.random.permutation(n_samples)
        
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        train_features = features_tensor[train_indices]
        val_features = features_tensor[val_indices]
        
        # Create datasets
        train_dataset = TensorDataset(train_features)
        val_dataset = TensorDataset(val_features)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        print(f"Data split: {len(train_features)} train, {len(val_features)} val")
        
        return train_loader, val_loader
    else:
        # No validation split
        dataset = TensorDataset(features_tensor)
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        print(f"Training on all {len(features)} samples")
        
        return train_loader, None
