"""
Evaluation Metrics Module
Compute clustering evaluation metrics.
"""

from typing import Dict, List
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)
import json
import os


def compute_cluster_purity(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    """
    Compute cluster purity metric.
    
    Cluster purity measures the fraction of the dominant class in each cluster.
    Higher values indicate better clustering quality.
    
    Args:
        true_labels: Ground truth labels
        predicted_labels: Predicted cluster labels
        
    Returns:
        Cluster purity score (0 to 1)
    """
    # Convert to numpy arrays and ensure integer type
    true_labels = np.asarray(true_labels, dtype=int)
    predicted_labels = np.asarray(predicted_labels, dtype=int)
    
    # Get unique clusters (filter out -1 for DBSCAN noise)
    clusters = np.unique(predicted_labels)
    clusters = clusters[clusters >= 0]  # Remove noise points
    
    total_correct = 0
    total_samples = 0
    
    for cluster in clusters:
        # Get samples in this cluster
        cluster_mask = predicted_labels == cluster
        cluster_true_labels = true_labels[cluster_mask]
        
        if len(cluster_true_labels) > 0:
            # Find most common true label in this cluster
            unique_labels, counts = np.unique(cluster_true_labels, return_counts=True)
            max_count = np.max(counts)
            total_correct += max_count
            total_samples += len(cluster_true_labels)
    
    if total_samples == 0:
        return 0.0
    
    purity = total_correct / total_samples
    return float(purity)


class ClusteringEvaluator:
    """
    Evaluate clustering performance using multiple metrics.
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.results = {}
        
    def evaluate_clustering(
        self,
        features: np.ndarray,
        predicted_labels: np.ndarray,
        true_labels: np.ndarray = None,
        method_name: str = "clustering"
    ) -> Dict[str, float]:
        """
        Evaluate clustering using multiple metrics.
        
        Args:
            features: Feature matrix used for clustering
            predicted_labels: Predicted cluster labels
            true_labels: Ground truth labels (optional, for supervised metrics)
            method_name: Name of clustering method
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Unsupervised metrics (no ground truth needed)
        try:
            metrics["silhouette_score"] = float(
                silhouette_score(features, predicted_labels)
            )
        except Exception as e:
            print(f"Warning: Could not compute silhouette score: {e}")
            metrics["silhouette_score"] = None
        
        try:
            metrics["davies_bouldin_index"] = float(
                davies_bouldin_score(features, predicted_labels)
            )
        except Exception as e:
            print(f"Warning: Could not compute Davies-Bouldin index: {e}")
            metrics["davies_bouldin_index"] = None
        
        try:
            metrics["calinski_harabasz_index"] = float(
                calinski_harabasz_score(features, predicted_labels)
            )
        except Exception as e:
            print(f"Warning: Could not compute Calinski-Harabasz index: {e}")
            metrics["calinski_harabasz_index"] = None
        
        # Supervised metrics (requires ground truth)
        if true_labels is not None:
            try:
                metrics["adjusted_rand_index"] = float(
                    adjusted_rand_score(true_labels, predicted_labels)
                )
            except Exception as e:
                print(f"Warning: Could not compute ARI: {e}")
                metrics["adjusted_rand_index"] = None
            
            try:
                metrics["normalized_mutual_info"] = float(
                    normalized_mutual_info_score(true_labels, predicted_labels)
                )
            except Exception as e:
                print(f"Warning: Could not compute NMI: {e}")
                metrics["normalized_mutual_info"] = None
            
            try:
                metrics["fowlkes_mallows_score"] = float(
                    fowlkes_mallows_score(true_labels, predicted_labels)
                )
            except Exception as e:
                print(f"Warning: Could not compute FMI: {e}")
                metrics["fowlkes_mallows_score"] = None
            
            try:
                metrics["cluster_purity"] = float(
                    compute_cluster_purity(true_labels, predicted_labels)
                )
            except Exception as e:
                print(f"Warning: Could not compute cluster purity: {e}")
                metrics["cluster_purity"] = None
            
            try:
                metrics["homogeneity_score"] = float(
                    homogeneity_score(true_labels, predicted_labels)
                )
            except Exception as e:
                print(f"Warning: Could not compute homogeneity: {e}")
                metrics["homogeneity_score"] = None
            
            try:
                metrics["completeness_score"] = float(
                    completeness_score(true_labels, predicted_labels)
                )
            except Exception as e:
                print(f"Warning: Could not compute completeness: {e}")
                metrics["completeness_score"] = None
            
            try:
                metrics["v_measure_score"] = float(
                    v_measure_score(true_labels, predicted_labels)
                )
            except Exception as e:
                print(f"Warning: Could not compute V-measure: {e}")
                metrics["v_measure_score"] = None
        
        self.results[method_name] = metrics
        return metrics
    
    def evaluate_multiple_methods(
        self,
        features: np.ndarray,
        clustering_results: Dict[str, np.ndarray],
        true_labels: np.ndarray = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate multiple clustering methods.
        
        Args:
            features: Feature matrix
            clustering_results: Dictionary of method_name -> cluster_labels
            true_labels: Ground truth labels (optional)
            
        Returns:
            Dictionary of evaluation results for each method
        """
        print("Evaluating clustering methods...")
        
        for method_name, predicted_labels in clustering_results.items():
            print(f"\nEvaluating {method_name}...")
            metrics = self.evaluate_clustering(
                features=features,
                predicted_labels=predicted_labels,
                true_labels=true_labels,
                method_name=method_name
            )
            self._print_metrics(method_name, metrics)
        
        return self.results
    
    def _print_metrics(self, method_name: str, metrics: Dict[str, float]):
        """Print metrics in a formatted way."""
        print(f"\n{method_name.upper()} Metrics:")
        print("-" * 50)
        
        # Unsupervised metrics
        if metrics.get("silhouette_score") is not None:
            print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
        
        if metrics.get("davies_bouldin_index") is not None:
            print(f"  Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f} (lower is better)")
        
        if metrics.get("calinski_harabasz_index") is not None:
            print(f"  Calinski-Harabasz Index: {metrics['calinski_harabasz_index']:.2f}")
        
        # Supervised metrics
        if metrics.get("adjusted_rand_index") is not None:
            print(f"  Adjusted Rand Index: {metrics['adjusted_rand_index']:.4f}")
        
        if metrics.get("normalized_mutual_info") is not None:
            print(f"  Normalized Mutual Info: {metrics['normalized_mutual_info']:.4f}")
        
        if metrics.get("fowlkes_mallows_score") is not None:
            print(f"  Fowlkes-Mallows Index: {metrics['fowlkes_mallows_score']:.4f}")
        
        if metrics.get("cluster_purity") is not None:
            print(f"  Cluster Purity: {metrics['cluster_purity']:.4f}")
        
        if metrics.get("homogeneity_score") is not None:
            print(f"  Homogeneity Score: {metrics['homogeneity_score']:.4f}")
        
        if metrics.get("completeness_score") is not None:
            print(f"  Completeness Score: {metrics['completeness_score']:.4f}")
        
        if metrics.get("v_measure_score") is not None:
            print(f"  V-Measure Score: {metrics['v_measure_score']:.4f}")
    
    def save_results(self, save_path: str):
        """
        Save evaluation results to JSON.
        
        Args:
            save_path: Path to save results
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nEvaluation results saved to {save_path}")
    
    def load_results(self, load_path: str):
        """
        Load evaluation results from JSON.
        
        Args:
            load_path: Path to load results from
        """
        with open(load_path, 'r') as f:
            self.results = json.load(f)
        
        print(f"Evaluation results loaded from {load_path}")
    
    def compare_methods(self) -> Dict[str, str]:
        """
        Compare methods and identify best performing.
        
        Returns:
            Dictionary of metric_name -> best_method
        """
        if not self.results:
            print("No results to compare")
            return {}
        
        comparison = {}
        
        # Get all metrics
        all_metrics = set()
        for metrics in self.results.values():
            all_metrics.update(metrics.keys())
        
        # Find best for each metric
        for metric in all_metrics:
            scores = {}
            for method, metrics in self.results.items():
                if metric in metrics and metrics[metric] is not None:
                    scores[method] = metrics[metric]
            
            if scores:
                # For Davies-Bouldin, lower is better
                if "davies_bouldin" in metric.lower():
                    best_method = min(scores, key=scores.get)
                else:
                    best_method = max(scores, key=scores.get)
                
                comparison[metric] = best_method
        
        return comparison
    
    def print_comparison(self):
        """Print comparison of methods."""
        comparison = self.compare_methods()
        
        if not comparison:
            return
        
        print("\n" + "=" * 60)
        print("BEST METHODS BY METRIC")
        print("=" * 60)
        
        for metric, method in comparison.items():
            score = self.results[method][metric]
            print(f"{metric:.<40} {method} ({score:.4f})")


def create_evaluation_summary(
    evaluator: ClusteringEvaluator,
    clustering_results: Dict[str, np.ndarray],
    true_labels: np.ndarray = None
) -> Dict:
    """
    Create a comprehensive evaluation summary.
    
    Args:
        evaluator: ClusteringEvaluator instance
        clustering_results: Dictionary of clustering results
        true_labels: Ground truth labels (optional)
        
    Returns:
        Summary dictionary
    """
    summary = {
        "methods": list(clustering_results.keys()),
        "metrics": evaluator.results,
        "best_by_metric": evaluator.compare_methods()
    }
    
    # Add cluster statistics
    from clustering import get_cluster_statistics
    
    summary["cluster_statistics"] = {}
    for method, labels in clustering_results.items():
        stats = get_cluster_statistics(labels)
        summary["cluster_statistics"][method] = stats
    
    return summary


if __name__ == "__main__":
    # Test evaluation
    print("Testing clustering evaluation...")
    
    # Create dummy data
    n_samples = 1000
    n_features = 64
    n_clusters = 10
    
    features = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Create dummy clustering results
    predicted_labels_1 = np.random.randint(0, n_clusters, n_samples)
    predicted_labels_2 = np.random.randint(0, n_clusters, n_samples)
    
    # Create dummy ground truth
    true_labels = np.random.randint(0, n_clusters, n_samples)
    
    clustering_results = {
        "method_1": predicted_labels_1,
        "method_2": predicted_labels_2
    }
    
    # Evaluate
    evaluator = ClusteringEvaluator()
    results = evaluator.evaluate_multiple_methods(
        features=features,
        clustering_results=clustering_results,
        true_labels=true_labels
    )
    
    # Print comparison
    evaluator.print_comparison()
    
    # Create summary
    summary = create_evaluation_summary(
        evaluator=evaluator,
        clustering_results=clustering_results,
        true_labels=true_labels
    )
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Methods evaluated: {', '.join(summary['methods'])}")
