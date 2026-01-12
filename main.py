"""
Main Pipeline
Complete end-to-end pipeline for unsupervised music clustering.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dataset import GTZANDataset, validate_gtzan_structure
from audio_features import AudioFeatureExtractor
from whisper_transcribe import WhisperTranscriber
from lyric_embedding import LyricEmbedder
from vae import create_vae
from advanced_vae import create_advanced_vae, ConditionalVAE
from train_vae import VAETrainer, prepare_data_loaders
from clustering import Clusterer
from evaluation import ClusteringEvaluator, create_evaluation_summary
from visualization import (
    visualize_all_results,
    visualize_vae_reconstructions,
    plot_genre_cluster_distribution,
    plot_cluster_purity_analysis
)


class MusicClusteringPipeline:
    """
    Complete pipeline for unsupervised music clustering.
    """
    
    def __init__(self, config: dict):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.results = {}
        
        # Create output directories
        os.makedirs(config['output_dir'], exist_ok=True)
        os.makedirs(config['cache_dir'], exist_ok=True)
        
    def run(self):
        """Run complete pipeline."""
        print("=" * 70)
        print("UNSUPERVISED MUSIC CLUSTERING PIPELINE")
        print("=" * 70)
        
        start_time = time.time()
        
        # Step 1: Load dataset
        print("\n[STEP 1/9] Loading GTZAN Dataset")
        print("-" * 70)
        audio_files, genre_labels = self.load_dataset()
        
        # Step 2: Extract audio features
        print("\n[STEP 2/9] Extracting Audio Features")
        print("-" * 70)
        audio_features = self.extract_audio_features(audio_files)
        
        # Step 3: Transcribe lyrics
        print("\n[STEP 3/9] Transcribing Lyrics with Whisper")
        print("-" * 70)
        self.transcribe_lyrics(audio_files)
        
        # Step 4: Generate lyric embeddings
        print("\n[STEP 4/9] Generating Lyric Embeddings")
        print("-" * 70)
        lyric_embeddings = self.generate_lyric_embeddings(audio_files)
        
        # Step 5: Fuse features
        print("\n[STEP 5/9] Fusing Audio and Lyric Features")
        print("-" * 70)
        fused_features = self.fuse_features(audio_features, lyric_embeddings)
        
        # Step 6: Train VAE
        print("\n[STEP 6/9] Training Variational Autoencoder")
        print("-" * 70)
        latent_features = self.train_vae(fused_features)
        
        # Step 7: Perform clustering
        print("\n[STEP 7/9] Performing Clustering")
        print("-" * 70)
        clustering_results = self.perform_clustering(latent_features)
        
        # Step 8: Evaluate
        print("\n[STEP 8/9] Evaluating Clustering Results")
        print("-" * 70)
        evaluation_results = self.evaluate_clustering(
            latent_features,
            clustering_results,
            genre_labels
        )
        
        # Step 9: Visualize
        print("\n[STEP 9/9] Generating Visualizations")
        print("-" * 70)
        self.visualize_results(
            latent_features,
            clustering_results,
            genre_labels,
            audio_files
        )
        
        # Save final results
        self.save_results(evaluation_results)
        
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"PIPELINE COMPLETED IN {elapsed_time:.2f} SECONDS")
        print("=" * 70)
        
    def load_dataset(self):
        """Load GTZAN dataset."""
        # Validate structure
        if not validate_gtzan_structure(self.config['data_dir']):
            raise RuntimeError("GTZAN dataset not found or invalid structure")
        
        # Load dataset
        dataset = GTZANDataset(self.config['data_dir'])
        audio_files, genre_labels = dataset.load_dataset()
        
        # Print statistics
        stats = dataset.get_statistics()
        print(f"Total audio files: {stats['total_files']}")
        print(f"Number of genres: {stats['num_genres']}")
        
        self.results['dataset'] = {
            'audio_files': audio_files,
            'genre_labels': genre_labels,
            'genre_names': dataset.genres
        }
        
        return audio_files, genre_labels
    
    def extract_audio_features(self, audio_files):
        """Extract audio features."""
        cache_path = os.path.join(self.config['cache_dir'], 'audio_features.pkl')
        
        extractor = AudioFeatureExtractor(
            sample_rate=self.config['audio']['sample_rate'],
            n_mfcc=self.config['audio']['n_mfcc'],
            n_mels=self.config['audio']['n_mels']
        )
        
        features = extractor.extract_features_batch(
            audio_files=audio_files,
            cache_path=cache_path
        )
        
        self.results['audio_features'] = features
        return features
    
    def transcribe_lyrics(self, audio_files):
        """Transcribe lyrics using Whisper."""
        transcriber = WhisperTranscriber(
            model_size=self.config['whisper']['model_size']
        )
        
        transcriptions = transcriber.transcribe_batch(
            audio_files=audio_files,
            output_dir=self.config['lyrics_dir'],
            force_retranscribe=self.config['whisper']['force_retranscribe']
        )
        
        # Print statistics
        stats = transcriber.get_transcription_statistics(transcriptions)
        print(f"Transcribed: {stats['with_lyrics']} songs with lyrics")
        print(f"Instrumental/Failed: {stats['instrumental_or_failed']} songs")
        
        self.results['transcriptions'] = transcriptions
        return transcriptions
    
    def generate_lyric_embeddings(self, audio_files):
        """Generate lyric embeddings."""
        cache_path = os.path.join(self.config['cache_dir'], 'lyric_embeddings.pkl')
        
        embedder = LyricEmbedder(
            model_name=self.config['embedding']['model_name']
        )
        
        # Generate embeddings
        track_ids = [embedder._get_track_id(path) for path in audio_files]
        embedding_dict = embedder.embed_from_files(
            lyrics_dir=self.config['lyrics_dir'],
            track_ids=track_ids,
            cache_path=cache_path
        )
        
        # Align with audio files
        embeddings = embedder.align_embeddings_with_audio_files(
            audio_files=audio_files,
            embedding_dict=embedding_dict
        )
        
        self.results['lyric_embeddings'] = embeddings
        return embeddings
    
    def fuse_features(self, audio_features, lyric_embeddings):
        """Fuse audio and lyric features."""
        print(f"Audio features shape: {audio_features.shape}")
        print(f"Lyric embeddings shape: {lyric_embeddings.shape}")
        
        # Concatenate features
        fused = np.concatenate([audio_features, lyric_embeddings], axis=1)
        
        # Normalize
        scaler = StandardScaler()
        fused_normalized = scaler.fit_transform(fused)
        
        print(f"Fused features shape: {fused_normalized.shape}")
        
        self.results['fused_features'] = fused_normalized
        self.results['scaler'] = scaler
        
        return fused_normalized
    
    def train_vae(self, features):
        """Train VAE model."""
        # Create model
        model = create_vae(
            input_dim=features.shape[1],
            latent_dim=self.config['vae']['latent_dim'],
            hidden_dims=self.config['vae']['hidden_dims'],
            dropout=self.config['vae']['dropout'],
            beta=self.config['vae']['beta']
        )
        
        print(f"VAE Architecture:")
        print(f"  Input dim: {features.shape[1]}")
        print(f"  Hidden dims: {self.config['vae']['hidden_dims']}")
        print(f"  Latent dim: {self.config['vae']['latent_dim']}")
        
        # Prepare data loaders
        train_loader, val_loader = prepare_data_loaders(
            features=features,
            batch_size=self.config['vae']['batch_size'],
            val_split=self.config['vae']['val_split']
        )
        
        # Create trainer
        trainer = VAETrainer(
            model=model,
            learning_rate=self.config['vae']['learning_rate'],
            beta=self.config['vae']['beta']
        )
        
        # Train
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.config['vae']['epochs'],
            verbose=True
        )
        
        # Save model
        model_path = os.path.join(self.config['output_dir'], 'vae_model.pt')
        trainer.save_model(model_path)
        
        # Plot training history
        history_plot_path = os.path.join(
            self.config['output_dir'],
            'vae_training_history.png'
        )
        trainer.plot_training_history(save_path=history_plot_path)
        
        # Get latent representations
        all_loader = prepare_data_loaders(
            features=features,
            batch_size=self.config['vae']['batch_size'],
            val_split=0.0,
            shuffle=False
        )[0]
        
        latent_features = trainer.get_latent_representations(all_loader)
        
        print(f"Latent features shape: {latent_features.shape}")
        
        self.results['vae_trainer'] = trainer
        self.results['latent_features'] = latent_features
        
        return latent_features
    
    def perform_clustering(self, features):
        """Perform clustering on latent features."""
        clusterer = Clusterer(random_state=self.config['random_state'])
        
        results = clusterer.cluster_all(
            features=features,
            n_clusters=self.config['clustering']['n_clusters'],
            pca_components=self.config['clustering'].get('pca_components', 100),
            include_dbscan=self.config['clustering'].get('include_dbscan', True),
            include_spectral=self.config['clustering'].get('include_spectral', True),
            include_autoencoder=self.config['clustering'].get('include_autoencoder', True)
        )
        
        # Print statistics
        from clustering import get_cluster_statistics
        for method, labels in results.items():
            stats = get_cluster_statistics(labels)
            print(f"\n{method.upper()}:")
            print(f"  Clusters: {stats['n_clusters']}")
            print(f"  Cluster sizes: {stats['min_size']}-{stats['max_size']} "
                  f"(mean: {stats['mean_size']:.1f})")
        
        # Save results
        clusterer.save_results(os.path.join(self.config['output_dir'], 'clustering'))
        
        self.results['clustering_results'] = results
        self.results['clusterer'] = clusterer
        
        return results
    
    def evaluate_clustering(self, features, clustering_results, true_labels):
        """Evaluate clustering results."""
        evaluator = ClusteringEvaluator()
        
        # Evaluate all methods
        evaluation_results = evaluator.evaluate_multiple_methods(
            features=features,
            clustering_results=clustering_results,
            true_labels=true_labels
        )
        
        # Print comparison
        evaluator.print_comparison()
        
        # Save results
        results_path = os.path.join(
            self.config['output_dir'],
            'evaluation_results.json'
        )
        evaluator.save_results(results_path)
        
        # Create summary
        summary = create_evaluation_summary(
            evaluator=evaluator,
            clustering_results=clustering_results,
            true_labels=true_labels
        )
        
        summary_path = os.path.join(
            self.config['output_dir'],
            'evaluation_summary.json'
        )
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.results['evaluation'] = evaluation_results
        self.results['evaluator'] = evaluator
        
        return evaluation_results
    
    def visualize_results(
        self,
        features,
        clustering_results,
        true_labels,
        audio_files
    ):
        """Generate visualizations."""
        genre_names = self.results['dataset']['genre_names']
        
        # VAE reconstruction visualization
        print("\nGenerating VAE reconstruction visualizations...")
        vae_recon_path = os.path.join(self.config['output_dir'], 'vae_reconstructions.png')
        visualize_vae_reconstructions(
            vae_model=self.results['vae_trainer'],
            features=self.results['fused_features'],
            n_samples=5,
            save_path=vae_recon_path
        )
        
        # Genre/cluster distribution analysis
        print("Generating genre-cluster distribution analysis...")
        for method_name, labels in clustering_results.items():
            # Skip DBSCAN if it has noise points
            if method_name == 'dbscan' and -1 in labels:
                continue
            
            dist_path = os.path.join(
                self.config['output_dir'],
                f'{method_name}_genre_distribution.png'
            )
            plot_genre_cluster_distribution(
                true_labels=true_labels,
                predicted_labels=labels,
                genre_names=genre_names,
                save_path=dist_path
            )
            
            purity_path = os.path.join(
                self.config['output_dir'],
                f'{method_name}_purity_analysis.png'
            )
            plot_cluster_purity_analysis(
                true_labels=true_labels,
                predicted_labels=labels,
                genre_names=genre_names,
                save_path=purity_path
            )
        
        # Visualize with both t-SNE and UMAP
        for method in ['tsne', 'umap']:
            print(f"\nGenerating {method.upper()} visualizations...")
            output_dir = os.path.join(self.config['output_dir'], f'plots_{method}')
            
            visualize_all_results(
                features=features,
                clustering_results=clustering_results,
                true_labels=true_labels,
                genre_names=genre_names,
                output_dir=output_dir,
                reduction_method=method
            )
    
    def save_results(self, evaluation_results):
        """Save final results summary."""
        summary = {
            "configuration": self.config,
            "dataset_statistics": {
                "total_files": len(self.results['dataset']['audio_files']),
                "genres": self.results['dataset']['genre_names']
            },
            "feature_dimensions": {
                "audio_features": self.results['audio_features'].shape[1],
                "lyric_embeddings": self.results['lyric_embeddings'].shape[1],
                "fused_features": self.results['fused_features'].shape[1],
                "latent_features": self.results['latent_features'].shape[1]
            },
            "evaluation_results": evaluation_results,
            "best_methods": self.results['evaluator'].compare_methods()
        }
        
        summary_path = os.path.join(self.config['output_dir'], 'pipeline_summary.json')
        with open(summary_path, 'w') as f:
            # Convert numpy types to native Python types
            json.dump(summary, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x))
        
        print(f"\nPipeline summary saved to {summary_path}")


def create_default_config():
    """Create default configuration."""
    return {
        # Directories
        'data_dir': 'data/audio/gtzan',
        'lyrics_dir': 'data/lyrics',
        'cache_dir': 'data/cache',
        'output_dir': 'outputs',
        
        # Audio feature extraction
        'audio': {
            'sample_rate': 22050,
            'n_mfcc': 20,
            'n_mels': 128
        },
        
        # Whisper transcription
        'whisper': {
            'model_size': 'base',
            'force_retranscribe': False
        },
        
        # Lyric embedding
        'embedding': {
            'model_name': 'all-MiniLM-L6-v2'
        },
        
        # VAE training
        'vae': {
            'latent_dim': 64,
            'hidden_dims': [256, 128],
            'dropout': 0.2,
            'beta': 1.0,
            'learning_rate': 1e-3,
            'batch_size': 32,
            'epochs': 25,
            'val_split': 0.2
        },
        
        # Clustering
        'clustering': {
            'n_clusters': 10,
            'pca_components': 50,
            'include_dbscan': True,
            'include_spectral': True,
            'include_autoencoder': True
        },
        
        # General
        'random_state': 42
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unsupervised Music Clustering Pipeline"
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/audio/gtzan',
        help='Path to GTZAN dataset directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory for results'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=25,
        help='Number of VAE training epochs'
    )
    parser.add_argument(
        '--latent-dim',
        type=int,
        default=64,
        help='VAE latent dimension'
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=10,
        help='Number of clusters'
    )
    parser.add_argument(
        '--whisper-model',
        type=str,
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model size'
    )
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # Override with command line arguments
    config['data_dir'] = args.data_dir
    config['output_dir'] = args.output_dir
    config['vae']['epochs'] = args.epochs
    config['vae']['latent_dim'] = args.latent_dim
    config['clustering']['n_clusters'] = args.n_clusters
    config['whisper']['model_size'] = args.whisper_model
    
    # Run pipeline
    pipeline = MusicClusteringPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
