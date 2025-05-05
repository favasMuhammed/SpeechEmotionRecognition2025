"""
Preprocess RAVDESS and EMODB datasets for Speech Emotion Recognition.
"""

import os
import numpy as np
import librosa
import json
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
import multiprocessing as mp
from datetime import datetime
from src.utils.model_config import (
    MODEL_CONFIG, FEATURE_CONFIG, AUDIO_CONFIG, EMOTIONS,
    EMOTION_MAPPINGS, RAVDESS_EMOTION_MAP, EMODB_EMOTION_MAP,
    DATASET_PATHS
)
from src.utils.logging_utils import setup_logger
from src.utils.audio_utils import (
    load_audio_file, extract_advanced_features,
    augment_audio, preprocess_audio
)

# Setup logger
logger = setup_logger('preprocess_data', 'preprocess')

def validate_directory(directory: str) -> bool:
    """
    Recursively validate if directory exists and contains WAV files in any subdirectory.
    Args:
        directory: Path to directory
    Returns:
        bool: True if directory is valid
    """
    if not os.path.exists(directory):
        logger.error(f"Directory not found: {directory}")
        return False
    wav_files = []
    for root, _, files in os.walk(directory):
        wav_files.extend([os.path.join(root, f) for f in files if f.endswith('.wav')])
    if not wav_files:
        logger.error(f"No WAV files found in {directory}")
        return False
    logger.info(f"Found {len(wav_files)} WAV files in {os.path.basename(directory)} dataset (recursive)")
    return True

def save_validation_stats(stats: Dict[str, Any], output_dir: str) -> None:
    """
    Save validation statistics to a JSON file.

    Args:
        stats: Dictionary containing validation statistics
        output_dir: Directory to save the statistics file
    """
    try:
        # Convert numpy types to Python native types
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj

        # Convert all numpy types in stats
        stats = convert_numpy_types(stats)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save statistics to JSON file
        stats_file = os.path.join(output_dir, 'validation_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=4)
        logger.info(f"Validation statistics saved to {stats_file}")
    except Exception as e:
        logger.error(f"Error saving validation statistics: {str(e)}")
        raise

def validate_audio_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate audio file meets requirements.

    Args:
        file_path: Path to audio file

    Returns:
        Tuple of (is_valid, reason)
    """
    try:
        # Load and validate audio
        audio, sr = load_audio_file(file_path)
        if audio is None or sr is None:
            return False, "Failed to load audio"
        
        # Check duration
        duration = len(audio) / sr
        if duration < AUDIO_CONFIG['min_duration'] or duration > AUDIO_CONFIG['max_duration']:
            return False, f"Duration {duration:.3f}s outside range [{AUDIO_CONFIG['min_duration']}-{AUDIO_CONFIG['max_duration']}]s"
        
        # Check amplitude
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude < AUDIO_CONFIG['min_amplitude']:
            return False, f"Amplitude {max_amplitude:.6f} below threshold {AUDIO_CONFIG['min_amplitude']}"
        
        return True, ""
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def process_audio_file(args: Tuple[str, str, Dict]) -> Optional[Tuple[np.ndarray, str]]:
    """
    Process a single audio file.

    Args:
        args: Tuple of (file_path, dataset_name, emotion_map)
        
    Returns:
        Tuple of (features, emotion) or None if processing failed
    """
    file_path, dataset_name, emotion_map = args
    
    try:
        # Validate file
        is_valid, reason = validate_audio_file(file_path)
        if not is_valid:
            logger.warning(f"Skipping {file_path}: {reason}")
            return None
        
        # Load audio
        audio, sr = load_audio_file(file_path)
        if audio is None or sr is None:
            logger.warning(f"Skipping {file_path}: Failed to load audio")
            return None
        
        # Normalize audio
        audio = librosa.util.normalize(audio)
        
        # Log audio properties
        duration = len(audio) / sr
        logger.debug(f"Processing {file_path}: length={len(audio)} samples, duration={duration:.3f}s, sample_rate={sr}")
        
        # Extract features
        features = preprocess_audio(audio, sr)
        
        # Get emotion label
        if dataset_name == 'RAVDESS':
            emotion_code = int(os.path.basename(file_path).split('-')[2])
            emotion = emotion_map.get(emotion_code, 'unknown')
        else:  # EMODB
            emotion_code = os.path.basename(file_path)[5]
            emotion = emotion_map.get(emotion_code, 'unknown')
        
        if emotion == 'unknown':
            logger.warning(f"Skipping {file_path}: Unknown emotion code {emotion_code}")
            return None
        
        return features, emotion
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return None

def augment_samples(features: np.ndarray, emotion: str, n_samples: int) -> List[Tuple[np.ndarray, str]]:
    """
    Generate augmented samples for a given feature vector.
    
    Args:
        features: Original feature vector
        emotion: Emotion label
        n_samples: Number of augmented samples to generate
        
    Returns:
        List of (augmented_features, emotion) tuples
    """
    augmented_samples = []
    for _ in range(n_samples):
        # Apply random augmentation to features
        augmented = features.copy()
        
        # Add small random noise
        noise = np.random.normal(0, 0.01, features.shape)
        augmented += noise
        
        # Random scaling
        scale = np.random.uniform(0.9, 1.1)
        augmented *= scale
        
        augmented_samples.append((augmented, emotion))
    
    return augmented_samples

def balance_dataset(features: List[np.ndarray], labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance the dataset using augmentation for underrepresented classes.
    
    Args:
        features: List of feature vectors
        labels: List of emotion labels
        
    Returns:
        Tuple of (balanced_features, balanced_labels)
    """
    # Count samples per class
    unique_labels, counts = np.unique(labels, return_counts=True)
    max_count = np.max(counts)
    
    balanced_features = []
    balanced_labels = []
    
    for label in unique_labels:
        # Get indices for current class
        indices = np.where(np.array(labels) == label)[0]
        class_features = [features[i] for i in indices]
        
        # Add original samples
        balanced_features.extend(class_features)
        balanced_labels.extend([label] * len(class_features))
        
        # Calculate number of augmented samples needed
        n_augment = max_count - len(indices)
        if n_augment > 0:
            # Generate augmented samples
            for feature in class_features:
                augmented = augment_samples(feature, label, n_augment // len(indices) + 1)
                for aug_feature, aug_label in augmented:
                    balanced_features.append(aug_feature)
                    balanced_labels.append(aug_label)
    
    return np.array(balanced_features), np.array(balanced_labels)

def main():
    """Main preprocessing function."""
    try:
        # Validate directories
        ravdess_dir = DATASET_PATHS['RAVDESS']
        emodb_dir = os.path.join(DATASET_PATHS['EMODB'], 'wav')
        if not validate_directory(ravdess_dir):
            return
        if not validate_directory(emodb_dir):
            return

        # Preliminary dataset integrity check
        logger.info("Checking dataset integrity...")
        ravdess_files = []
        for root, _, files in os.walk(ravdess_dir):
            for file in files:
                if file.endswith('.wav'):
                    ravdess_files.append(os.path.join(root, file))
        
        emodb_files = []
        for root, _, files in os.walk(emodb_dir):
            for file in files:
                if file.endswith('.wav'):
                    emodb_files.append(os.path.join(root, file))
        
        # Validate a sample of files
        sample_size = min(10, len(ravdess_files), len(emodb_files))
        for dataset_name, files in [('RAVDESS', ravdess_files[:sample_size]), ('EMODB', emodb_files[:sample_size])]:
            logger.info(f"Validating sample of {dataset_name} files...")
            for file_path in files:
                is_valid, reason = validate_audio_file(file_path)
                if not is_valid:
                    logger.warning(f"Sample file invalid: {file_path} - {reason}")
        
        # Process files in parallel
        logger.info("Processing RAVDESS dataset...")
        with mp.Pool(processes=mp.cpu_count()) as pool:
            ravdess_args = [(f, 'RAVDESS', RAVDESS_EMOTION_MAP) for f in ravdess_files]
            ravdess_results = list(tqdm(
                pool.imap(process_audio_file, ravdess_args),
                total=len(ravdess_files),
                desc="Processing RAVDESS"
            ))
            
            logger.info("Processing EMODB dataset...")
            emodb_args = [(f, 'EMODB', EMODB_EMOTION_MAP) for f in emodb_files]
            emodb_results = list(tqdm(
                pool.imap(process_audio_file, emodb_args),
                total=len(emodb_files),
                desc="Processing EMODB"
            ))
        
        # Combine results
        all_results = [r for r in ravdess_results + emodb_results if r is not None]
        if not all_results:
            logger.error("No valid audio files processed")
            return
        
        features = [r[0] for r in all_results]
        labels = [r[1] for r in all_results]
        
        # Balance dataset
        if MODEL_CONFIG['training']['class_weights']:
            features, labels = balance_dataset(features, labels)
        
        # Convert to numpy arrays
        features = np.array(features)
        labels = np.array(labels)
        
        # Save processed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        features_dir = os.path.join(DATASET_PATHS['processed'], 'features')
        labels_dir = os.path.join(DATASET_PATHS['processed'], 'labels')
        
        os.makedirs(features_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        np.save(os.path.join(features_dir, f'features_{timestamp}.npy'), features)
        np.save(os.path.join(labels_dir, f'labels_{timestamp}.npy'), labels)
        
        # Save validation statistics
        stats = {
            'total_samples': len(features),
            'feature_shape': features.shape,
            'emotion_distribution': dict(zip(*np.unique(labels, return_counts=True))),
            'timestamp': timestamp,
            'skipped_files': len(ravdess_files) + len(emodb_files) - len(all_results),
            'dataset_stats': {
                'RAVDESS': {
                    'total_files': len(ravdess_files),
                    'processed_files': len([r for r in ravdess_results if r is not None]),
                    'skipped_files': len([r for r in ravdess_results if r is None])
                },
                'EMODB': {
                    'total_files': len(emodb_files),
                    'processed_files': len([r for r in emodb_results if r is not None]),
                    'skipped_files': len([r for r in emodb_results if r is None])
                }
            }
        }
        save_validation_stats(stats, DATASET_PATHS['processed'])
        
        logger.info(f"Preprocessing completed. Processed {len(features)} samples, skipped {stats['skipped_files']} files.")
        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Emotion distribution: {stats['emotion_distribution']}")
        logger.info(f"Dataset statistics: {stats['dataset_stats']}")
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main()