"""
Lyric Embedding Generation
Generate embeddings for transcribed lyrics using SentenceTransformer.
"""

import os
from pathlib import Path
from typing import Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import joblib
import torch


class LyricEmbedder:
    """
    Generate embeddings for lyrics using SentenceTransformer.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = None
    ):
        """
        Initialize lyric embedder.
        
        Args:
            model_name: SentenceTransformer model name
            device: Device to run on (cuda/cpu). Auto-detect if None.
        """
        self.model_name = model_name
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading SentenceTransformer model: {model_name} on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        # Handle special cases
        if not text or text in ["[INSTRUMENTAL]", "[ERROR]"]:
            # Return zero vector for instrumental/error cases
            return np.zeros(self.embedding_dim)
        
        # Generate embedding
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        return embedding
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for encoding
            
        Returns:
            Embedding matrix (n_samples, embedding_dim)
        """
        print(f"Generating embeddings for {len(texts)} texts...")
        
        # Process texts
        processed_texts = []
        for text in texts:
            if not text or text in ["[INSTRUMENTAL]", "[ERROR]"]:
                # Use placeholder text for special cases
                processed_texts.append("no lyrics available")
            else:
                processed_texts.append(text)
        
        # Generate embeddings
        embeddings = self.model.encode(
            processed_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        print(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def embed_from_files(
        self,
        lyrics_dir: str = "data/lyrics",
        track_ids: List[str] = None,
        cache_path: str = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings from lyric files with caching.
        
        Args:
            lyrics_dir: Directory containing lyric files
            track_ids: List of track IDs to process (None for all)
            cache_path: Path to cache embeddings (optional)
            
        Returns:
            Dictionary mapping track_id to embedding
        """
        # Check cache
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached lyric embeddings from {cache_path}")
            return joblib.load(cache_path)
        
        # Load lyrics
        lyrics_path = Path(lyrics_dir)
        
        if not lyrics_path.exists():
            print(f"Error: Lyrics directory not found: {lyrics_dir}")
            return {}
        
        # Get all lyric files
        txt_files = list(lyrics_path.glob("*.txt"))
        
        if not txt_files:
            print(f"Error: No lyric files found in {lyrics_dir}")
            return {}
        
        # Filter by track_ids if provided
        if track_ids:
            txt_files = [f for f in txt_files if f.stem in track_ids]
        
        # Load texts
        track_id_list = []
        text_list = []
        
        for txt_file in tqdm(txt_files, desc="Loading lyrics"):
            track_id = txt_file.stem
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            track_id_list.append(track_id)
            text_list.append(text)
        
        # Generate embeddings
        embeddings = self.embed_batch(text_list)
        
        # Create dictionary
        embedding_dict = {
            track_id: embedding
            for track_id, embedding in zip(track_id_list, embeddings)
        }
        
        # Cache embeddings
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            joblib.dump(embedding_dict, cache_path)
            print(f"Cached lyric embeddings to {cache_path}")
        
        return embedding_dict
    
    def align_embeddings_with_audio_files(
        self,
        audio_files: List[str],
        embedding_dict: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Align lyric embeddings with audio file order.
        
        Args:
            audio_files: List of audio file paths
            embedding_dict: Dictionary of embeddings
            
        Returns:
            Embedding matrix aligned with audio_files
        """
        embeddings = []
        
        for audio_path in audio_files:
            # Generate track ID
            track_id = self._get_track_id(audio_path)
            
            # Get embedding or use zero vector
            if track_id in embedding_dict:
                embedding = embedding_dict[track_id]
            else:
                print(f"Warning: No embedding found for {track_id}, using zero vector")
                embedding = np.zeros(self.embedding_dim)
            
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def _get_track_id(self, audio_path: str) -> str:
        """
        Generate track ID from audio path.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Track ID (genre_filename)
        """
        path = Path(audio_path)
        genre = path.parent.name
        filename = path.stem
        return f"{genre}_{filename}"


def generate_lyric_embeddings(
    audio_files: List[str],
    lyrics_dir: str = "data/lyrics",
    cache_path: str = "data/embeddings/lyric_embeddings.pkl",
    model_name: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Convenience function to generate lyric embeddings for dataset.
    
    Args:
        audio_files: List of audio file paths
        lyrics_dir: Directory containing lyric files
        cache_path: Path to cache embeddings
        model_name: SentenceTransformer model name
        
    Returns:
        Embedding matrix (n_samples, embedding_dim)
    """
    # Initialize embedder
    embedder = LyricEmbedder(model_name=model_name)
    
    # Generate track IDs
    track_ids = [embedder._get_track_id(path) for path in audio_files]
    
    # Generate embeddings
    embedding_dict = embedder.embed_from_files(
        lyrics_dir=lyrics_dir,
        track_ids=track_ids,
        cache_path=cache_path
    )
    
    # Align with audio files
    embeddings = embedder.align_embeddings_with_audio_files(
        audio_files=audio_files,
        embedding_dict=embedding_dict
    )
    
    return embeddings


if __name__ == "__main__":
    # Test embedding generation
    import sys
    
    if len(sys.argv) > 1:
        # Test single text
        test_text = " ".join(sys.argv[1:])
        embedder = LyricEmbedder()
        embedding = embedder.embed_text(test_text)
        print(f"Input text: {test_text}")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
    else:
        # Test with lyrics directory
        lyrics_dir = "data/lyrics"
        if os.path.exists(lyrics_dir):
            embedder = LyricEmbedder()
            embedding_dict = embedder.embed_from_files(
                lyrics_dir=lyrics_dir,
                cache_path="data/embeddings/test_embeddings.pkl"
            )
            print(f"\nGenerated embeddings for {len(embedding_dict)} tracks")
            
            # Show sample
            if embedding_dict:
                sample_id = list(embedding_dict.keys())[0]
                sample_embedding = embedding_dict[sample_id]
                print(f"\nSample embedding ({sample_id}):")
                print(f"  Shape: {sample_embedding.shape}")
                print(f"  Norm: {np.linalg.norm(sample_embedding):.4f}")
                print(f"  Mean: {np.mean(sample_embedding):.4f}")
                print(f"  Std: {np.std(sample_embedding):.4f}")
        else:
            print(f"Lyrics directory not found: {lyrics_dir}")
            print("Please run whisper_transcribe.py first")
