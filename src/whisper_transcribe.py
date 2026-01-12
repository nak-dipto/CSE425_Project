"""
Whisper Lyrics Transcription
Transcribe lyrics from audio files using OpenAI Whisper.
"""

import os
from pathlib import Path
from typing import List, Dict
import whisper
import torch
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class WhisperTranscriber:
    """
    Transcribe audio to text using OpenAI Whisper.
    """
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = None,
        language: str = "en"
    ):
        """
        Initialize Whisper transcriber.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (cuda/cpu). Auto-detect if None.
            language: Language code (en for English)
        """
        self.model_size = model_size
        self.language = language
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading Whisper {model_size} model on {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)
        print("Whisper model loaded successfully")
        
    def transcribe_audio(
        self,
        audio_path: str,
        verbose: bool = False
    ) -> Dict[str, any]:
        """
        Transcribe a single audio file.
        
        Args:
            audio_path: Path to audio file
            verbose: Print transcription progress
            
        Returns:
            Dictionary with transcription results
        """
        try:
            # Transcribe with Whisper
            result = self.model.transcribe(
                audio_path,
                language=self.language,
                verbose=verbose,
                task="transcribe"
            )
            
            # Extract text
            text = result.get("text", "").strip()
            
            # Check if transcription is empty or too short
            if len(text) < 10:
                # Likely instrumental or failed transcription
                text = "[INSTRUMENTAL]"
            
            return {
                "text": text,
                "language": result.get("language", self.language),
                "success": True
            }
            
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
            return {
                "text": "[ERROR]",
                "language": self.language,
                "success": False,
                "error": str(e)
            }
    
    def transcribe_batch(
        self,
        audio_files: List[str],
        output_dir: str = "data/lyrics",
        force_retranscribe: bool = False
    ) -> Dict[str, str]:
        """
        Transcribe multiple audio files and save to text files.
        
        Args:
            audio_files: List of audio file paths
            output_dir: Directory to save transcriptions
            force_retranscribe: Re-transcribe even if file exists
            
        Returns:
            Dictionary mapping track_id to transcription text
        """
        os.makedirs(output_dir, exist_ok=True)
        transcriptions = {}
        
        print(f"Transcribing {len(audio_files)} audio files with Whisper...")
        
        for audio_path in tqdm(audio_files, desc="Transcribing"):
            # Generate track ID
            track_id = self._get_track_id(audio_path)
            output_path = os.path.join(output_dir, f"{track_id}.txt")
            
            # Check if already transcribed
            if os.path.exists(output_path) and not force_retranscribe:
                with open(output_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                transcriptions[track_id] = text
                continue
            
            # Transcribe
            result = self.transcribe_audio(audio_path, verbose=False)
            text = result["text"]
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            transcriptions[track_id] = text
        
        print(f"Transcription complete. Saved to {output_dir}/")
        return transcriptions
    
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
    
    def load_transcriptions(
        self,
        lyrics_dir: str = "data/lyrics"
    ) -> Dict[str, str]:
        """
        Load existing transcriptions from directory.
        
        Args:
            lyrics_dir: Directory containing transcription files
            
        Returns:
            Dictionary mapping track_id to transcription text
        """
        transcriptions = {}
        lyrics_path = Path(lyrics_dir)
        
        if not lyrics_path.exists():
            print(f"Lyrics directory not found: {lyrics_dir}")
            return transcriptions
        
        txt_files = list(lyrics_path.glob("*.txt"))
        
        for txt_file in txt_files:
            track_id = txt_file.stem
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            transcriptions[track_id] = text
        
        print(f"Loaded {len(transcriptions)} transcriptions from {lyrics_dir}")
        return transcriptions
    
    def get_transcription_statistics(
        self,
        transcriptions: Dict[str, str]
    ) -> Dict:
        """
        Get statistics about transcriptions.
        
        Args:
            transcriptions: Dictionary of transcriptions
            
        Returns:
            Dictionary with statistics
        """
        if not transcriptions:
            return {}
        
        texts = list(transcriptions.values())
        lengths = [len(text.split()) for text in texts]
        
        instrumental_count = sum(
            1 for text in texts
            if "[INSTRUMENTAL]" in text or "[ERROR]" in text or len(text) < 20
        )
        
        stats = {
            "total": len(texts),
            "instrumental_or_failed": instrumental_count,
            "with_lyrics": len(texts) - instrumental_count,
            "avg_word_count": sum(lengths) / len(lengths) if lengths else 0,
            "min_word_count": min(lengths) if lengths else 0,
            "max_word_count": max(lengths) if lengths else 0
        }
        
        return stats


def transcribe_dataset(
    audio_files: List[str],
    model_size: str = "base",
    output_dir: str = "data/lyrics",
    force_retranscribe: bool = False
) -> Dict[str, str]:
    """
    Convenience function to transcribe entire dataset.
    
    Args:
        audio_files: List of audio file paths
        model_size: Whisper model size
        output_dir: Output directory for transcriptions
        force_retranscribe: Re-transcribe existing files
        
    Returns:
        Dictionary of transcriptions
    """
    transcriber = WhisperTranscriber(model_size=model_size)
    transcriptions = transcriber.transcribe_batch(
        audio_files=audio_files,
        output_dir=output_dir,
        force_retranscribe=force_retranscribe
    )
    
    # Print statistics
    stats = transcriber.get_transcription_statistics(transcriptions)
    print("\nTranscription Statistics:")
    print(f"  Total: {stats.get('total', 0)}")
    print(f"  With lyrics: {stats.get('with_lyrics', 0)}")
    print(f"  Instrumental/Failed: {stats.get('instrumental_or_failed', 0)}")
    print(f"  Avg word count: {stats.get('avg_word_count', 0):.1f}")
    
    return transcriptions


if __name__ == "__main__":
    # Test transcription
    import sys
    from dataset import GTZANDataset
    
    if len(sys.argv) > 1:
        # Transcribe single file
        test_audio = sys.argv[1]
        if os.path.exists(test_audio):
            transcriber = WhisperTranscriber(model_size="base")
            result = transcriber.transcribe_audio(test_audio, verbose=True)
            print(f"\nTranscription: {result['text']}")
        else:
            print(f"Audio file not found: {test_audio}")
    else:
        # Transcribe small batch for testing
        print("Loading dataset for transcription test...")
        dataset = GTZANDataset()
        audio_files, _ = dataset.load_dataset()
        
        # Test with first 5 files
        test_files = audio_files[:5]
        print(f"\nTranscribing {len(test_files)} test files...")
        
        transcribe_dataset(
            audio_files=test_files,
            model_size="base",
            output_dir="data/lyrics"
        )
