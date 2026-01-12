"""
GTZAN Dataset Loader
Loads audio files from GTZAN Genre Collection and extracts genre labels.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm


class GTZANDataset:
    """
    GTZAN Genre Collection dataset loader.
    Assumes directory structure: data/audio/gtzan/genre_name/*.wav
    """
    
    def __init__(self, data_dir: str = "data/audio/gtzan"):
        """
        Initialize GTZAN dataset loader.
        
        Args:
            data_dir: Path to GTZAN dataset directory
        """
        self.data_dir = Path(data_dir)
        self.genres = [
            "blues", "classical", "country", "disco", "hiphop",
            "jazz", "metal", "pop", "reggae", "rock"
        ]
        self.audio_files = []
        self.labels = []
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(self.genres)}
        
    def load_dataset(self) -> Tuple[List[str], List[int]]:
        """
        Load all audio file paths and their corresponding genre labels.
        
        Returns:
            Tuple of (audio_file_paths, genre_labels)
        """
        print("Loading GTZAN dataset...")
        
        for genre in self.genres:
            genre_dir = self.data_dir / genre
            
            if not genre_dir.exists():
                print(f"Warning: Genre directory not found: {genre_dir}")
                continue
                
            # Get all .wav and .au files
            wav_files = list(genre_dir.glob("*.wav"))
            au_files = list(genre_dir.glob("*.au"))
            files = wav_files + au_files
            
            if not files:
                print(f"Warning: No audio files found in {genre_dir}")
                continue
            
            for audio_file in files:
                self.audio_files.append(str(audio_file))
                self.labels.append(self.genre_to_idx[genre])
        
        print(f"Loaded {len(self.audio_files)} audio files from {len(set(self.labels))} genres")
        
        return self.audio_files, self.labels
    
    def get_genre_name(self, label_idx: int) -> str:
        """Get genre name from label index."""
        return self.genres[label_idx]
    
    def get_track_id(self, audio_path: str) -> str:
        """
        Extract track ID from audio file path.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Track ID (genre_filename)
        """
        path = Path(audio_path)
        genre = path.parent.name
        filename = path.stem
        return f"{genre}_{filename}"
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self.audio_files:
            self.load_dataset()
        
        stats = {
            "total_files": len(self.audio_files),
            "num_genres": len(set(self.labels)),
            "genres": self.genres,
            "files_per_genre": {}
        }
        
        for genre in self.genres:
            genre_idx = self.genre_to_idx[genre]
            count = self.labels.count(genre_idx)
            stats["files_per_genre"][genre] = count
        
        return stats


def validate_gtzan_structure(data_dir: str = "data/audio/gtzan") -> bool:
    """
    Validate that GTZAN dataset has correct directory structure.
    
    Args:
        data_dir: Path to GTZAN dataset directory
        
    Returns:
        True if structure is valid, False otherwise
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Dataset directory does not exist: {data_dir}")
        print("Please download GTZAN dataset and extract to data/audio/gtzan/")
        return False
    
    genres = [
        "blues", "classical", "country", "disco", "hiphop",
        "jazz", "metal", "pop", "reggae", "rock"
    ]
    
    missing_genres = []
    empty_genres = []
    
    for genre in genres:
        genre_dir = data_path / genre
        if not genre_dir.exists():
            missing_genres.append(genre)
        else:
            wav_files = list(genre_dir.glob("*.wav"))
            au_files = list(genre_dir.glob("*.au"))
            if not wav_files and not au_files:
                empty_genres.append(genre)
    
    if missing_genres:
        print(f"Warning: Missing genre directories: {missing_genres}")
    
    if empty_genres:
        print(f"Warning: Empty genre directories: {empty_genres}")
    
    if missing_genres or empty_genres:
        print("\nExpected structure:")
        print("data/audio/gtzan/")
        print("├── blues/*.wav")
        print("├── classical/*.wav")
        print("├── country/*.wav")
        print("├── disco/*.wav")
        print("├── hiphop/*.wav")
        print("├── jazz/*.wav")
        print("├── metal/*.wav")
        print("├── pop/*.wav")
        print("├── reggae/*.wav")
        print("└── rock/*.wav")
        return False
    
    return True


if __name__ == "__main__":
    # Test dataset loading
    if validate_gtzan_structure():
        dataset = GTZANDataset()
        audio_files, labels = dataset.load_dataset()
        stats = dataset.get_statistics()
        
        print("\nDataset Statistics:")
        print(f"Total files: {stats['total_files']}")
        print(f"Number of genres: {stats['num_genres']}")
        print("\nFiles per genre:")
        for genre, count in stats['files_per_genre'].items():
            print(f"  {genre}: {count}")
