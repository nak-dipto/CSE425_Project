"""
Visualization Module
Visualize clustering results and latent space representations.
"""

import os
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from matplotlib.patches import Patch
import torch


def visualize_vae_reconstructions(
    vae_model,
    features: np.ndarray,
    n_samples: int = 5,
    save_path: str = None
):
    """
    Visualize VAE reconstruction quality.
    
    Args:
        vae_model: Trained VAE model or VAETrainer
        features: Original features
        n_samples: Number of samples to visualize
        save_path: Path to save plot
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    
    # Handle VAETrainer vs raw model
    if hasattr(vae_model, 'model'):
        model = vae_model.model
        device = vae_model.device
    else:
        model = vae_model
        device = next(model.parameters()).device
    
    # Select random samples
    indices = np.random.choice(len(features), n_samples, replace=False)
    samples = features[indices]
    
    # Get reconstructions
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(samples).to(device)
        recon, _, _ = model(x)
        recon = recon.cpu().numpy()
    
    # Plot
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3 * n_samples))
    
    for i in range(n_samples):
        # Original
        axes[i, 0].plot(samples[i])
        axes[i, 0].set_title(f"Original Sample {i+1}")
        axes[i, 0].set_ylabel("Feature Value")
        axes[i, 0].grid(True, alpha=0.3)
        
        # Reconstruction
        axes[i, 1].plot(recon[i])
        axes[i, 1].set_title(f"Reconstruction {i+1}")
        axes[i, 1].set_ylabel("Feature Value")
        axes[i, 1].grid(True, alpha=0.3)
    
    axes[-1, 0].set_xlabel("Feature Dimension")
    axes[-1, 1].set_xlabel("Feature Dimension")
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reconstruction plot saved to {save_path}")
    
    plt.close()


def plot_genre_cluster_distribution(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    genre_names: List[str],
    save_path: str = None
):
    """
    Plot distribution of genres across clusters.
    
    Args:
        true_labels: True genre labels
        predicted_labels: Predicted cluster labels
        genre_names: Names of genres
        save_path: Path to save plot
    """
    # Convert to numpy arrays with proper dtype
    true_labels = np.asarray(true_labels, dtype=int)
    predicted_labels = np.asarray(predicted_labels, dtype=int)
    
    # Filter out noise points from DBSCAN (label = -1)
    valid_mask = predicted_labels >= 0
    true_labels = true_labels[valid_mask]
    predicted_labels = predicted_labels[valid_mask]
    
    # Get unique values
    unique_clusters = np.unique(predicted_labels)
    n_clusters = len(unique_clusters)
    n_genres = len(np.unique(true_labels))
    
    # Create distribution matrix
    dist_matrix = np.zeros((n_genres, n_clusters))
    
    for genre_idx in range(n_genres):
        genre_mask = true_labels == genre_idx
        genre_clusters = predicted_labels[genre_mask]
        
        for cluster_idx_pos, cluster_id in enumerate(unique_clusters):
            count = np.sum(genre_clusters == cluster_id)
            dist_matrix[genre_idx, cluster_idx_pos] = count
    
    # Normalize by row (genre), avoiding division by zero
    row_sums = dist_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    dist_matrix_norm = dist_matrix / row_sums
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(dist_matrix_norm, cmap='YlOrRd', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(n_clusters))
    ax.set_yticks(np.arange(n_genres))
    ax.set_xticklabels([f"C{cid}" for cid in unique_clusters])
    ax.set_yticklabels(genre_names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Proportion of Genre in Cluster", rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(n_genres):
        for j in range(n_clusters):
            count = int(dist_matrix[i, j])
            prop = dist_matrix_norm[i, j]
            # Handle NaN values
            if np.isnan(prop):
                prop = 0.0
            text = ax.text(
                j, i, f'{count}\n({prop:.2f})',
                ha="center", va="center",
                color="white" if prop > 0.5 else "black",
                fontsize=8
            )
    
    ax.set_xlabel("Cluster ID", fontsize=12)
    ax.set_ylabel("True Genre", fontsize=12)
    ax.set_title("Genre Distribution Across Clusters", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Genre distribution plot saved to {save_path}")
    
    plt.close()


def plot_cluster_purity_analysis(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    genre_names: List[str],
    save_path: str = None
):
    """
    Analyze and visualize cluster purity.
    
    Args:
        true_labels: True genre labels
        predicted_labels: Predicted cluster labels
        genre_names: Names of genres
        save_path: Path to save plot
    """
    # Convert to numpy arrays with proper dtype
    true_labels = np.asarray(true_labels, dtype=int)
    predicted_labels = np.asarray(predicted_labels, dtype=int)
    
    clusters = np.unique(predicted_labels)
    # Filter out noise points for DBSCAN
    clusters = clusters[clusters >= 0]
    
    purity_scores = []
    dominant_genres = []
    cluster_sizes = []
    
    for cluster_idx in clusters:
        mask = predicted_labels == cluster_idx
        cluster_true_labels = true_labels[mask]
        
        if len(cluster_true_labels) > 0:
            unique, counts = np.unique(cluster_true_labels, return_counts=True)
            dominant_idx = unique[np.argmax(counts)]
            purity = np.max(counts) / len(cluster_true_labels)
            
            purity_scores.append(purity)
            dominant_genres.append(genre_names[dominant_idx])
            cluster_sizes.append(len(cluster_true_labels))
        else:
            purity_scores.append(0)
            dominant_genres.append("Empty")
            cluster_sizes.append(0)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Purity scores
    colors = plt.cm.RdYlGn(purity_scores)
    bars1 = ax1.bar(clusters, purity_scores, color=colors, edgecolor='black')
    
    for i, (cluster, purity, genre) in enumerate(zip(clusters, purity_scores, dominant_genres)):
        ax1.text(cluster, purity + 0.02, f'{genre}\n({purity:.2f})', 
                ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel("Cluster ID", fontsize=12)
    ax1.set_ylabel("Purity Score", fontsize=12)
    ax1.set_title("Cluster Purity (Dominant Genre)", fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.15)
    ax1.grid(axis='y', alpha=0.3)
    
    # Cluster sizes
    ax2.bar(clusters, cluster_sizes, color='steelblue', edgecolor='black')
    
    for cluster, size in zip(clusters, cluster_sizes):
        ax2.text(cluster, size, f'{size}', ha='center', va='bottom', fontsize=10)
    
    ax2.set_xlabel("Cluster ID", fontsize=12)
    ax2.set_ylabel("Number of Samples", fontsize=12)
    ax2.set_title("Cluster Sizes", fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cluster purity analysis saved to {save_path}")
    
    plt.close()


class ClusterVisualizer:
    """
    Visualize clustering results and latent representations.
    """
    
    def __init__(self, figsize: tuple = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = figsize
        
    def plot_latent_space_2d(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        method: str = "tsne",
        title: str = "Latent Space Visualization",
        label_names: List[str] = None,
        save_path: str = None,
        show_legend: bool = True
    ):
        """
        Plot 2D projection of latent space.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Labels for coloring points
            method: Dimensionality reduction method (tsne or umap)
            title: Plot title
            label_names: Names for labels (optional)
            save_path: Path to save plot (optional)
            show_legend: Show legend
        """
        print(f"Computing 2D projection using {method.upper()}...")
        
        # Reduce to 2D
        if method.lower() == "tsne":
            reducer = TSNE(
                n_components=2,
                random_state=42,
                perplexity=min(30, len(features) - 1)
            )
            coords_2d = reducer.fit_transform(features)
        elif method.lower() == "umap":
            reducer = umap.UMAP(
                n_components=2,
                random_state=42,
                n_neighbors=min(15, len(features) - 1)
            )
            coords_2d = reducer.fit_transform(features)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Get unique labels
        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)
        
        # Create color map
        colors = plt.cm.tab20(np.linspace(0, 1, n_labels))
        
        # Plot each cluster
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            
            if label_names is not None and label < len(label_names):
                label_text = label_names[label]
            else:
                label_text = f"Cluster {label}"
            
            ax.scatter(
                coords_2d[mask, 0],
                coords_2d[mask, 1],
                c=[colors[idx]],
                label=label_text,
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidth=0.5
            )
        
        ax.set_xlabel(f"{method.upper()} Dimension 1", fontsize=12)
        ax.set_ylabel(f"{method.upper()} Dimension 2", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        if show_legend and n_labels <= 20:
            ax.legend(
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                frameon=True,
                fontsize=10
            )
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.close()
    
    def plot_multiple_clusterings(
        self,
        features: np.ndarray,
        clustering_results: Dict[str, np.ndarray],
        method: str = "tsne",
        save_dir: str = None
    ):
        """
        Plot multiple clustering results side by side.
        
        Args:
            features: Feature matrix
            clustering_results: Dictionary of method_name -> labels
            method: Dimensionality reduction method
            save_dir: Directory to save plots
        """
        n_methods = len(clustering_results)
        
        # Compute 2D projection once
        print(f"Computing 2D projection using {method.upper()}...")
        if method.lower() == "tsne":
            reducer = TSNE(
                n_components=2,
                random_state=42,
                perplexity=min(30, len(features) - 1)
            )
            coords_2d = reducer.fit_transform(features)
        elif method.lower() == "umap":
            reducer = umap.UMAP(
                n_components=2,
                random_state=42,
                n_neighbors=min(15, len(features) - 1)
            )
            coords_2d = reducer.fit_transform(features)
        
        # Create subplots
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        
        if n_methods == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot each method
        for idx, (method_name, labels) in enumerate(clustering_results.items()):
            ax = axes[idx]
            
            unique_labels = np.unique(labels)
            n_labels = len(unique_labels)
            colors = plt.cm.tab20(np.linspace(0, 1, n_labels))
            
            for label_idx, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(
                    coords_2d[mask, 0],
                    coords_2d[mask, 1],
                    c=[colors[label_idx]],
                    label=f"Cluster {label}",
                    alpha=0.6,
                    s=30,
                    edgecolors='black',
                    linewidth=0.3
                )
            
            ax.set_title(method_name.upper(), fontsize=12, fontweight='bold')
            ax.set_xlabel(f"{method.upper()} 1", fontsize=10)
            ax.set_ylabel(f"{method.upper()} 2", fontsize=10)
        
        # Hide unused subplots
        for idx in range(n_methods, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"clustering_comparison_{method}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.close()
    
    def plot_cluster_distribution(
        self,
        labels: np.ndarray,
        title: str = "Cluster Distribution",
        save_path: str = None
    ):
        """
        Plot distribution of samples across clusters.
        
        Args:
            labels: Cluster labels
            title: Plot title
            save_path: Path to save plot
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(unique_labels, counts, color='steelblue', edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        ax.set_xlabel("Cluster ID", fontsize=12)
        ax.set_ylabel("Number of Samples", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution plot saved to {save_path}")
        
        plt.close()
    
    def plot_confusion_matrix(
        self,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray,
        true_label_names: List[str] = None,
        title: str = "Cluster vs Genre Confusion Matrix",
        save_path: str = None
    ):
        """
        Plot confusion matrix between true labels and clusters.
        
        Args:
            true_labels: Ground truth labels
            predicted_labels: Predicted cluster labels
            true_label_names: Names for true labels
            title: Plot title
            save_path: Path to save plot
        """
        from sklearn.metrics import confusion_matrix
        
        # Compute confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Normalize by row (genre)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot heatmap
        im = ax.imshow(cm_normalized, cmap='YlOrRd', aspect='auto')
        
        # Set ticks
        n_true = len(np.unique(true_labels))
        n_pred = len(np.unique(predicted_labels))
        
        ax.set_xticks(np.arange(n_pred))
        ax.set_yticks(np.arange(n_true))
        
        if true_label_names is not None:
            ax.set_yticklabels(true_label_names)
        else:
            ax.set_yticklabels([f"Genre {i}" for i in range(n_true)])
        
        ax.set_xticklabels([f"C{i}" for i in range(n_pred)])
        
        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Proportion", rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(n_true):
            for j in range(n_pred):
                text = ax.text(
                    j, i, f'{cm[i, j]}',
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > 0.5 else "black",
                    fontsize=8
                )
        
        ax.set_xlabel("Cluster ID", fontsize=12)
        ax.set_ylabel("True Genre", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.close()


def visualize_all_results(
    features: np.ndarray,
    clustering_results: Dict[str, np.ndarray],
    true_labels: np.ndarray = None,
    genre_names: List[str] = None,
    output_dir: str = "outputs",
    reduction_method: str = "tsne"
):
    """
    Generate all visualizations for clustering results.
    
    Args:
        features: Feature matrix
        clustering_results: Dictionary of clustering results
        true_labels: Ground truth labels (optional)
        genre_names: Names for genres (optional)
        output_dir: Output directory for plots
        reduction_method: Method for dimensionality reduction
    """
    visualizer = ClusterVisualizer()
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    # 1. Individual clustering visualizations
    for method_name, labels in clustering_results.items():
        save_path = os.path.join(
            output_dir,
            f"{method_name}_{reduction_method}.png"
        )
        visualizer.plot_latent_space_2d(
            features=features,
            labels=labels,
            method=reduction_method,
            title=f"{method_name.upper()} Clustering",
            save_path=save_path
        )
    
    # 2. Comparison plot
    visualizer.plot_multiple_clusterings(
        features=features,
        clustering_results=clustering_results,
        method=reduction_method,
        save_dir=output_dir
    )
    
    # 3. Cluster distributions
    for method_name, labels in clustering_results.items():
        save_path = os.path.join(
            output_dir,
            f"{method_name}_distribution.png"
        )
        visualizer.plot_cluster_distribution(
            labels=labels,
            title=f"{method_name.upper()} Cluster Distribution",
            save_path=save_path
        )
    
    # 4. True genre visualization (if available)
    if true_labels is not None:
        save_path = os.path.join(
            output_dir,
            f"true_genres_{reduction_method}.png"
        )
        visualizer.plot_latent_space_2d(
            features=features,
            labels=true_labels,
            method=reduction_method,
            title="True Genre Labels",
            label_names=genre_names,
            save_path=save_path
        )
        
        # 5. Confusion matrices
        for method_name, labels in clustering_results.items():
            save_path = os.path.join(
                output_dir,
                f"{method_name}_confusion_matrix.png"
            )
            visualizer.plot_confusion_matrix(
                true_labels=true_labels,
                predicted_labels=labels,
                true_label_names=genre_names,
                title=f"{method_name.upper()} vs True Genres",
                save_path=save_path
            )
    
    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == "__main__":
    # Test visualization
    print("Testing visualization...")
    
    # Create dummy data
    n_samples = 500
    n_features = 64
    n_clusters = 10
    
    features = np.random.randn(n_samples, n_features).astype(np.float32)
    predicted_labels = np.random.randint(0, n_clusters, n_samples)
    true_labels = np.random.randint(0, n_clusters, n_samples)
    
    clustering_results = {
        "kmeans": predicted_labels,
        "agglomerative": np.random.randint(0, n_clusters, n_samples)
    }
    
    genre_names = [f"Genre_{i}" for i in range(n_clusters)]
    
    # Visualize
    visualize_all_results(
        features=features,
        clustering_results=clustering_results,
        true_labels=true_labels,
        genre_names=genre_names,
        output_dir="outputs/test",
        reduction_method="tsne"
    )
    
    print("Visualization test complete!")
