"""
Clustering Module
Perform clustering on VAE latent representations.
"""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import os


class Clusterer:
    """
    Clustering algorithms for latent representations.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize clusterer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.labels = {}
        
    def kmeans_clustering(
        self,
        features: np.ndarray,
        n_clusters: int = 10,
        n_init: int = 50,
        max_iter: int = 500
    ) -> np.ndarray:
        """
        Perform K-Means clustering with improved initialization.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            n_clusters: Number of clusters
            n_init: Number of initializations
            max_iter: Maximum iterations
            
        Returns:
            Cluster labels
        """
        print(f"Performing K-Means clustering (k={n_clusters}, n_init={n_init})...")
        
        # L2 normalize features for better angular separation
        from sklearn.preprocessing import normalize
        features_normalized = normalize(features, norm='l2')
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            algorithm='lloyd',
            random_state=self.random_state,
            tol=1e-5
        )
        
        labels = kmeans.fit_predict(features_normalized)
        
        self.models["kmeans"] = kmeans
        self.labels["kmeans"] = labels
        
        print(f"K-Means clustering complete (inertia: {kmeans.inertia_:.2f})")
        return labels
    
    def agglomerative_clustering(
        self,
        features: np.ndarray,
        n_clusters: int = 10,
        linkage: str = "ward"
    ) -> np.ndarray:
        """
        Perform Agglomerative clustering with normalization.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            n_clusters: Number of clusters
            linkage: Linkage criterion (ward, complete, average, single)
            
        Returns:
            Cluster labels
        """
        print(f"Performing Agglomerative clustering (k={n_clusters}, linkage={linkage})...")
        
        # L2 normalize for better separation
        from sklearn.preprocessing import normalize
        features_normalized = normalize(features, norm='l2')
        
        agg = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
        
        labels = agg.fit_predict(features_normalized)
        
        self.models["agglomerative"] = agg
        self.labels["agglomerative"] = labels
        
        print(f"Agglomerative clustering complete")
        return labels
    
    def dbscan_clustering(
        self,
        features: np.ndarray,
        eps: float = None,
        min_samples: int = 3,
        eps_percentile: int = 75
    ) -> np.ndarray:
        """
        Perform DBSCAN clustering with automatic epsilon tuning.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            eps: Maximum distance (auto-computed if None)
            min_samples: Minimum samples in neighborhood
            eps_percentile: Percentile for automatic eps computation
            
        Returns:
            Cluster labels (-1 for noise points)
        """
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
        
        # Normalize features for better distance computation
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        
        # Auto-compute eps if not provided
        if eps is None:
            neighbors = NearestNeighbors(n_neighbors=min_samples)
            neighbors.fit(features_normalized)
            distances, _ = neighbors.kneighbors(features_normalized)
            eps = np.percentile(distances[:, -1], eps_percentile)
            print(f"Auto-computed eps={eps:.4f} (percentile={eps_percentile})")
        
        print(f"Performing DBSCAN clustering (eps={eps:.4f}, min_samples={min_samples})...")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = dbscan.fit_predict(features_normalized)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Calculate cluster statistics
        if n_clusters > 0:
            cluster_ids = [i for i in set(labels) if i != -1]
            sizes = [np.sum(labels == i) for i in cluster_ids]
            print(f"  Found {n_clusters} clusters")
            print(f"  Cluster sizes: {min(sizes)}-{max(sizes)} (mean: {np.mean(sizes):.1f})")
            print(f"  Noise points: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
        else:
            print(f"  Found {n_clusters} clusters")
            print(f"  Noise points: {n_noise}")
        
        self.models["dbscan"] = dbscan
        self.models["dbscan_scaler"] = scaler
        self.labels["dbscan"] = labels
        
        return labels
    
    def spectral_clustering_direct(
        self,
        features: np.ndarray,
        n_clusters: int = 10
    ) -> np.ndarray:
        """
        Direct clustering on spectral features without VAE.
        
        Args:
            features: Raw feature matrix
            n_clusters: Number of clusters
            
        Returns:
            Cluster labels
        """
        from sklearn.cluster import SpectralClustering
        
        print(f"Performing Spectral clustering (k={n_clusters})...")
        
        # L2 normalize for better graph construction
        from sklearn.preprocessing import normalize
        features_normalized = normalize(features, norm='l2')
        
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            random_state=self.random_state,
            affinity='nearest_neighbors',
            n_neighbors=15,
            assign_labels='kmeans',
            n_init=20
        )
        
        labels = spectral.fit_predict(features_normalized)
        
        self.models["spectral"] = spectral
        self.labels["spectral"] = labels
        
        print(f"Spectral clustering complete")
        return labels
    
    def autoencoder_baseline(
        self,
        features: np.ndarray,
        n_clusters: int = 10,
        latent_dim: int = 64
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Baseline: Standard Autoencoder + K-Means.
        
        Args:
            features: Feature matrix
            n_clusters: Number of clusters
            latent_dim: Autoencoder latent dimension
            
        Returns:
            Tuple of (cluster_labels, autoencoder_features)
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        
        print(f"Training standard Autoencoder baseline...")
        
        # Simple autoencoder
        class SimpleAutoencoder(nn.Module):
            def __init__(self, input_dim, latent_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, latent_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Linear(512, input_dim)
                )
            
            def forward(self, x):
                z = self.encoder(x)
                recon = self.decoder(z)
                return recon, z
        
        # Train autoencoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SimpleAutoencoder(features.shape[1], latent_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Prepare data
        dataset = TensorDataset(torch.FloatTensor(features))
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train for 50 epochs
        model.train()
        for epoch in range(25):
            total_loss = 0
            for batch in loader:
                x = batch[0].to(device)
                recon, z = model(x)
                loss = nn.functional.mse_loss(recon, x)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/50, Loss: {total_loss/len(loader):.4f}")
        
        # Extract features
        model.eval()
        ae_features = []
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=32, shuffle=False):
                x = batch[0].to(device)
                _, z = model(x)
                ae_features.append(z.cpu().numpy())
        
        ae_features = np.vstack(ae_features)
        
        # K-Means on AE features
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=self.random_state)
        labels = kmeans.fit_predict(ae_features)
        
        self.models["autoencoder"] = model
        self.models["ae_kmeans"] = kmeans
        self.labels["ae_kmeans"] = labels
        
        print(f"Autoencoder + K-Means baseline complete")
        return labels, ae_features
    
    def pca_kmeans_baseline(
        self,
        features: np.ndarray,
        n_clusters: int = 10,
        n_components: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Baseline: PCA + K-Means clustering.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            n_clusters: Number of clusters
            n_components: Number of PCA components
            
        Returns:
            Tuple of (cluster_labels, pca_features)
        """
        print(f"Performing PCA + K-Means baseline (k={n_clusters}, pca={n_components})...")
        
        # PCA dimensionality reduction
        pca = PCA(n_components=n_components, random_state=self.random_state)
        pca_features = pca.fit_transform(features)
        
        explained_var = np.sum(pca.explained_variance_ratio_)
        print(f"PCA explained variance: {explained_var:.4f}")
        
        # K-Means on PCA features
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            max_iter=300,
            random_state=self.random_state
        )
        
        labels = kmeans.fit_predict(pca_features)
        
        self.models["pca"] = pca
        self.models["pca_kmeans"] = kmeans
        self.labels["pca_kmeans"] = labels
        
        print(f"PCA + K-Means baseline complete")
        return labels, pca_features
    
    def cluster_all(
        self,
        features: np.ndarray,
        n_clusters: int = 10,
        pca_components: int = 50,
        include_dbscan: bool = True,
        include_spectral: bool = True,
        include_autoencoder: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Perform all clustering methods.
        
        Args:
            features: Feature matrix
            n_clusters: Number of clusters
            pca_components: Number of PCA components for baseline
            include_dbscan: Whether to include DBSCAN
            include_spectral: Whether to include Spectral clustering
            include_autoencoder: Whether to include Autoencoder baseline
            
        Returns:
            Dictionary of clustering results
        """
        results = {}
        
        # K-Means
        results["kmeans"] = self.kmeans_clustering(features, n_clusters=n_clusters)
        
        # Agglomerative
        results["agglomerative"] = self.agglomerative_clustering(
            features, n_clusters=n_clusters
        )
        
        # PCA + K-Means baseline
        results["pca_kmeans"], pca_features = self.pca_kmeans_baseline(
            features, n_clusters=n_clusters, n_components=pca_components
        )
        
        # DBSCAN (if requested)
        if include_dbscan:
            results["dbscan"] = self.dbscan_clustering(
                features, eps=None, min_samples=3, eps_percentile=75
            )
        
        # Spectral clustering (if requested)
        if include_spectral:
            try:
                results["spectral"] = self.spectral_clustering_direct(
                    features, n_clusters=n_clusters
                )
            except Exception as e:
                print(f"Spectral clustering failed: {e}")
        
        # Autoencoder baseline (if requested)
        if include_autoencoder:
            try:
                results["ae_kmeans"], _ = self.autoencoder_baseline(
                    features, n_clusters=n_clusters, latent_dim=64
                )
            except Exception as e:
                print(f"Autoencoder baseline failed: {e}")
        
        return results
    
    def save_results(self, save_dir: str):
        """
        Save clustering models and labels.
        
        Args:
            save_dir: Directory to save results
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save models (skip unpicklable ones like PyTorch models)
        for name, model in self.models.items():
            # Skip PyTorch models and autoencoder - they need special handling
            if name == 'autoencoder' or 'torch' in str(type(model).__module__):
                continue
            
            model_path = os.path.join(save_dir, f"{name}_model.pkl")
            try:
                joblib.dump(model, model_path)
            except Exception as e:
                print(f"  Warning: Could not save {name} model: {e}")
                continue
        
        # Save labels
        labels_path = os.path.join(save_dir, "cluster_labels.pkl")
        joblib.dump(self.labels, labels_path)
        
        print(f"Clustering results saved to {save_dir}")
    
    def load_results(self, save_dir: str):
        """
        Load clustering results.
        
        Args:
            save_dir: Directory with saved results
        """
        # Load labels
        labels_path = os.path.join(save_dir, "cluster_labels.pkl")
        if os.path.exists(labels_path):
            self.labels = joblib.load(labels_path)
            print(f"Loaded clustering labels from {save_dir}")
        else:
            print(f"No clustering labels found in {save_dir}")


def get_cluster_statistics(labels: np.ndarray) -> Dict:
    """
    Get statistics about cluster assignments.
    
    Args:
        labels: Cluster labels
        
    Returns:
        Dictionary with statistics
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    stats = {
        "n_clusters": len(unique_labels),
        "cluster_sizes": dict(zip(unique_labels.tolist(), counts.tolist())),
        "min_size": int(np.min(counts)),
        "max_size": int(np.max(counts)),
        "mean_size": float(np.mean(counts)),
        "std_size": float(np.std(counts))
    }
    
    return stats


def find_optimal_k(
    features: np.ndarray,
    k_range: range = range(2, 21),
    method: str = "elbow"
) -> int:
    """
    Find optimal number of clusters using elbow method or silhouette.
    
    Args:
        features: Feature matrix
        k_range: Range of k values to test
        method: Method to use (elbow or silhouette)
        
    Returns:
        Optimal k value
    """
    from sklearn.metrics import silhouette_score
    
    scores = []
    
    print(f"Finding optimal k using {method} method...")
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(features)
        
        if method == "elbow":
            scores.append(kmeans.inertia_)
        elif method == "silhouette":
            score = silhouette_score(features, labels)
            scores.append(score)
    
    if method == "elbow":
        # Find elbow point (simple method: maximum second derivative)
        scores = np.array(scores)
        diffs = np.diff(scores)
        second_diffs = np.diff(diffs)
        optimal_idx = np.argmax(np.abs(second_diffs))
        optimal_k = list(k_range)[optimal_idx + 1]
    elif method == "silhouette":
        # Find maximum silhouette score
        optimal_idx = np.argmax(scores)
        optimal_k = list(k_range)[optimal_idx]
    
    print(f"Optimal k: {optimal_k}")
    return optimal_k


if __name__ == "__main__":
    # Test clustering
    print("Testing clustering algorithms...")
    
    # Create dummy data
    n_samples = 1000
    n_features = 64
    features = np.random.randn(n_samples, n_features).astype(np.float32)
    
    print(f"Feature shape: {features.shape}")
    
    # Initialize clusterer
    clusterer = Clusterer(random_state=42)
    
    # Perform all clustering methods
    results = clusterer.cluster_all(
        features,
        n_clusters=10,
        pca_components=32
    )
    
    # Print statistics
    print("\nClustering Statistics:")
    for method, labels in results.items():
        stats = get_cluster_statistics(labels)
        print(f"\n{method.upper()}:")
        print(f"  Number of clusters: {stats['n_clusters']}")
        print(f"  Cluster sizes: min={stats['min_size']}, max={stats['max_size']}, "
              f"mean={stats['mean_size']:.1f}, std={stats['std_size']:.1f}")
    
    # Test optimal k finding
    print("\nTesting optimal k finding...")
    optimal_k = find_optimal_k(features, k_range=range(2, 11), method="silhouette")
