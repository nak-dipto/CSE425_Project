"""
Utility script for common operations.
"""

import os
import sys
import shutil
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def clean_cache():
    """Clean cached data."""
    cache_dirs = ['data/cache', 'data/lyrics', 'data/embeddings']
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            print(f"Cleaning {cache_dir}...")
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir)
            print(f"  Cleaned!")
        else:
            print(f"  {cache_dir} does not exist")
    
    print("\nCache cleaned!")


def clean_outputs():
    """Clean output directory."""
    output_dir = 'outputs'
    
    if os.path.exists(output_dir):
        print(f"Cleaning {output_dir}...")
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        print("Outputs cleaned!")
    else:
        print(f"{output_dir} does not exist")


def validate_dataset():
    """Validate GTZAN dataset structure."""
    from dataset import validate_gtzan_structure, GTZANDataset
    
    print("Validating GTZAN dataset structure...")
    
    if validate_gtzan_structure():
        print("\nDataset structure is valid!")
        
        # Load and print statistics
        dataset = GTZANDataset()
        audio_files, labels = dataset.load_dataset()
        stats = dataset.get_statistics()
        
        print("\nDataset Statistics:")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Number of genres: {stats['num_genres']}")
        print("\n  Files per genre:")
        for genre, count in stats['files_per_genre'].items():
            print(f"    {genre:.<15} {count}")
    else:
        print("\nDataset structure is INVALID!")
        print("Please download GTZAN and extract to data/audio/gtzan/")


def test_audio_features():
    """Test audio feature extraction."""
    from dataset import GTZANDataset
    from audio_features import AudioFeatureExtractor
    
    print("Testing audio feature extraction...")
    
    # Load one file
    dataset = GTZANDataset()
    audio_files, _ = dataset.load_dataset()
    
    if not audio_files:
        print("No audio files found!")
        return
    
    test_file = audio_files[0]
    print(f"\nTesting with: {test_file}")
    
    # Extract features
    extractor = AudioFeatureExtractor()
    features = extractor.extract_all_features(test_file)
    
    print(f"\nFeature extraction successful!")
    print(f"  Feature shape: {features.shape}")
    print(f"  Feature range: [{features.min():.4f}, {features.max():.4f}]")


def test_whisper():
    """Test Whisper transcription."""
    from dataset import GTZANDataset
    from whisper_transcribe import WhisperTranscriber
    
    print("Testing Whisper transcription...")
    
    # Load one file
    dataset = GTZANDataset()
    audio_files, _ = dataset.load_dataset()
    
    if not audio_files:
        print("No audio files found!")
        return
    
    test_file = audio_files[0]
    print(f"\nTranscribing: {test_file}")
    
    # Transcribe
    transcriber = WhisperTranscriber(model_size="base")
    result = transcriber.transcribe_audio(test_file, verbose=False)
    
    print(f"\nTranscription successful!")
    print(f"  Text: {result['text'][:200]}...")
    print(f"  Language: {result['language']}")


def show_results():
    """Show results from latest run."""
    import json
    
    results_file = 'outputs/pipeline_summary.json'
    
    if not os.path.exists(results_file):
        print("No results found. Run the pipeline first!")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("=" * 70)
    print("PIPELINE RESULTS SUMMARY")
    print("=" * 70)
    
    print("\nDataset:")
    print(f"  Total files: {results['dataset_statistics']['total_files']}")
    print(f"  Genres: {', '.join(results['dataset_statistics']['genres'])}")
    
    print("\nFeature Dimensions:")
    for key, value in results['feature_dimensions'].items():
        print(f"  {key}: {value}")
    
    print("\nBest Methods:")
    for metric, method in results['best_methods'].items():
        print(f"  {metric}: {method}")
    
    print("\nEvaluation Results:")
    for method, metrics in results['evaluation_results'].items():
        print(f"\n  {method.upper()}:")
        for metric_name, value in metrics.items():
            if value is not None:
                print(f"    {metric_name}: {value:.4f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Utility script for music clustering")
    parser.add_argument(
        'command',
        choices=[
            'clean-cache',
            'clean-outputs',
            'validate',
            'test-audio',
            'test-whisper',
            'show-results'
        ],
        help='Command to execute'
    )
    
    args = parser.parse_args()
    
    if args.command == 'clean-cache':
        clean_cache()
    elif args.command == 'clean-outputs':
        clean_outputs()
    elif args.command == 'validate':
        validate_dataset()
    elif args.command == 'test-audio':
        test_audio_features()
    elif args.command == 'test-whisper':
        test_whisper()
    elif args.command == 'show-results':
        show_results()


if __name__ == "__main__":
    main()
