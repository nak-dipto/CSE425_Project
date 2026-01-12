"""
Audio Feature Extraction
Extract MFCC and Mel-Spectrogram features from audio files using librosa.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import joblib


class AudioFeatureExtractor:
    """
    Extract audio features (MFCC, Mel-Spectrogram) from audio files.
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_mfcc: int = 20,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        max_duration: float = 30.0
    ):
        """
        Initialize audio feature extractor.
        
        Args:
            sample_rate: Target sample rate for audio
            n_mfcc: Number of MFCC coefficients
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Hop length for STFT
            max_duration: Maximum audio duration in seconds (for truncation)
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_duration = max_duration
        self.max_length = int(sample_rate * max_duration)
        
    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load and preprocess audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Audio waveform as numpy array
        """
        try:
            # Load audio with librosa
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Truncate or pad to max_length
            if len(audio) > self.max_length:
                audio = audio[:self.max_length]
            elif len(audio) < self.max_length:
                audio = np.pad(audio, (0, self.max_length - len(audio)), mode='constant')
            
            return audio
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return silence if loading fails
            return np.zeros(self.max_length)
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio.
        
        Args:
            audio: Audio waveform
            
        Returns:
            MFCC features (flattened)
        """
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Compute statistics over time
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_delta = np.mean(librosa.feature.delta(mfccs), axis=1)
        
        # Concatenate features
        features = np.concatenate([mfcc_mean, mfcc_std, mfcc_delta])
        
        return features
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract Mel-Spectrogram features from audio.
        
        Args:
            audio: Audio waveform
            
        Returns:
            Mel-Spectrogram features (flattened)
        """
        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Compute statistics over time
        mel_mean = np.mean(mel_spec_db, axis=1)
        mel_std = np.std(mel_spec_db, axis=1)
        
        # Concatenate features
        features = np.concatenate([mel_mean, mel_std])
        
        return features
    
    def extract_spectral_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract additional spectral features.
        
        Args:
            audio: Audio waveform
            
        Returns:
            Spectral features (flattened)
        """
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=self.hop_length)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        # Compute statistics
        features = np.concatenate([
            np.mean(spectral_centroid, axis=1),
            np.std(spectral_centroid, axis=1),
            np.mean(spectral_rolloff, axis=1),
            np.std(spectral_rolloff, axis=1),
            np.mean(zcr, axis=1),
            np.std(zcr, axis=1),
            np.mean(chroma, axis=1),
            np.std(chroma, axis=1)
        ])
        
        return features
    
    def extract_all_features(self, audio_path: str) -> np.ndarray:
        """
        Extract all audio features from a file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Combined audio features
        """
        # Load audio
        audio = self.load_audio(audio_path)
        
        # Extract features
        mfcc_features = self.extract_mfcc(audio)
        mel_features = self.extract_mel_spectrogram(audio)
        spectral_features = self.extract_spectral_features(audio)
        
        # Concatenate all features
        all_features = np.concatenate([
            mfcc_features,
            mel_features,
            spectral_features
        ])
        
        return all_features
    
    def extract_features_batch(
        self,
        audio_files: List[str],
        cache_path: str = None
    ) -> np.ndarray:
        """
        Extract features from multiple audio files with caching.
        
        Args:
            audio_files: List of audio file paths
            cache_path: Path to cache features (optional)
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        # Check cache
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached audio features from {cache_path}")
            return joblib.load(cache_path)
        
        print("Extracting audio features...")
        features_list = []
        
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            features = self.extract_all_features(audio_file)
            features_list.append(features)
        
        features_matrix = np.array(features_list)
        
        # Cache features
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            joblib.dump(features_matrix, cache_path)
            print(f"Cached audio features to {cache_path}")
        
        print(f"Extracted audio features shape: {features_matrix.shape}")
        return features_matrix


def get_audio_info(audio_path: str) -> Dict:
    """
    Get audio file information.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with audio information
    """
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=audio, sr=sr)
        
        return {
            "duration": duration,
            "sample_rate": sr,
            "num_samples": len(audio),
            "channels": 1  # librosa loads as mono
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Test feature extraction
    import sys
    
    if len(sys.argv) > 1:
        test_audio = sys.argv[1]
    else:
        test_audio = "data/audio/gtzan/blues/blues.00000.wav"
    
    if os.path.exists(test_audio):
        print(f"Testing feature extraction on: {test_audio}")
        
        # Get audio info
        info = get_audio_info(test_audio)
        print(f"\nAudio info: {info}")
        
        # Extract features
        extractor = AudioFeatureExtractor()
        features = extractor.extract_all_features(test_audio)
        print(f"\nExtracted features shape: {features.shape}")
        print(f"Feature statistics:")
        print(f"  Mean: {np.mean(features):.4f}")
        print(f"  Std: {np.std(features):.4f}")
        print(f"  Min: {np.min(features):.4f}")
        print(f"  Max: {np.max(features):.4f}")
    else:
        print(f"Test audio file not found: {test_audio}")
        print("Please provide a valid audio file path or ensure GTZAN dataset is in data/audio/gtzan/")
