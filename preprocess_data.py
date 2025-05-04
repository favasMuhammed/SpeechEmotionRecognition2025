"""
Preprocess the RAVDESS and EMODB datasets for Speech Emotion Recognition.

This script loads audio files, validates their format and content, applies data
augmentation, extracts features, and saves the preprocessed data with versioning.
It includes validation statistics and logging for debugging.
"""

import os
import numpy as np
import librosa
from tqdm import tqdm
import logging
from datetime import datetime
from src.utils.model_config import DATASET_CONFIG, EMOTIONS, FEATURE_CONFIG, EMODB_EMOTION_MAP
from src.utils.audio_utils import extract_features, augment_audio
from src.utils.logging_utils import setup_logger

# Set up logger
logger = setup_logger('preprocess_data', 'preprocess')

def validate_audio(audio, file_path, sample_rate, duration):
    """
    Validate an audio signal's properties.

    Args:
        audio (numpy.ndarray): Audio signal.
        file_path (str): Path to the audio file (for logging).
        sample_rate (int): Expected sample rate in Hz.
        duration (float): Expected duration in seconds.

    Returns:
        bool: True if valid, False otherwise.
    """
    if len(audio) == 0:
        logging.warning(f"Empty audio file: {file_path}")
        return False
    
    expected_length = int(sample_rate * duration)
    actual_duration = len(audio) / sample_rate
    if actual_duration < duration * 0.5 or actual_duration > duration * 1.5:
        logging.warning(f"Audio duration out of range in {file_path}: {actual_duration:.2f}s (expected ~{duration}s)")
        return False
    
    if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
        logging.warning(f"Invalid audio values (NaN/Inf) in {file_path}")
        return False
    
    if np.max(np.abs(audio)) < 0.01:  # Threshold for silence
        logging.warning(f"Audio too quiet in {file_path}")
        return False
    
    return True

def validate_audio_file(file_path, expected_duration, expected_sr):
    """
    Validate an audio file's properties before processing.

    Args:
        file_path (str): Path to the audio file.
        expected_duration (float): Expected duration in seconds.
        expected_sr (int): Expected sample rate in Hz.

    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return False
        
        audio, sr = librosa.load(file_path, sr=None)
        if sr != expected_sr:
            logging.warning(f"Sample rate mismatch in {file_path}: expected {expected_sr}, got {sr}")
            return False
        
        return validate_audio(audio, file_path, expected_sr, expected_duration)
    except Exception as e:
        logging.error(f"Error validating {file_path}: {str(e)}")
        return False

def validate_emotion_label(emotion_id, dataset_name):
    """
    Validate the emotion label based on the dataset.

    Args:
        emotion_id (str or int): Emotion identifier (integer for RAVDESS, string for EMODB).
        dataset_name (str): Name of the dataset ('RAVDESS' or 'EMODB').

    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        if dataset_name == 'RAVDESS':
            emotion_id = int(emotion_id)
            if not 1 <= emotion_id <= 8:
                logging.warning(f"Invalid RAVDESS emotion ID: {emotion_id}. Expected 1-8 (neutral, calm, happy, sad, angry, fear, disgust, ps).")
                return False
        elif dataset_name == 'EMODB':
            if emotion_id not in EMODB_EMOTION_MAP:
                logging.warning(f"Invalid EMODB emotion code: {emotion_id}. Expected one of {list(EMODB_EMOTION_MAP.keys())}.")
                return False
        else:
            logging.error(f"Unknown dataset: {dataset_name}")
            return False
        return True
    except Exception as e:
        logging.error(f"Error validating emotion for {dataset_name}: {str(e)}")
        return False

def preprocess_dataset(ravdess_dir, emodb_dir):
    """
    Preprocess the RAVDESS and EMODB datasets with augmentation and validation.

    Args:
        ravdess_dir (str): Path to the RAVDESS dataset directory.
        emodb_dir (str): Path to the EMODB dataset directory.

    Saves:
        Preprocessed features and labels as NumPy arrays with versioning.
    """
    features = []
    labels = []
    validation_stats = {
        'total_files': 0,
        'valid_files': 0,
        'invalid_files': 0,
        'skipped_files': 0
    }
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Process RAVDESS dataset
    logging.info("Processing RAVDESS dataset...")
    emotion_counts = {emotion: 0 for emotion in EMOTIONS}
    for actor_folder in os.listdir(ravdess_dir):
        actor_path = os.path.join(ravdess_dir, actor_folder)
        if not os.path.isdir(actor_path):
            continue
        
        for audio_file in tqdm(os.listdir(actor_path), desc=f"Processing {actor_folder}"):
            validation_stats['total_files'] += 1
            if not audio_file.endswith('.wav'):
                validation_stats['skipped_files'] += 1
                continue
            
            try:
                # Extract and validate emotion ID
                parts = audio_file.split('-')
                if len(parts) < 3:
                    logging.warning(f"Invalid RAVDESS filename format: {audio_file}")
                    validation_stats['invalid_files'] += 1
                    continue
                emotion_id = int(parts[2])
                if not validate_emotion_label(emotion_id, 'RAVDESS'):
                    validation_stats['invalid_files'] += 1
                    continue
                
                emotion = EMOTIONS[emotion_id - 1]
                if emotion_counts[emotion] >= DATASET_CONFIG['samples_per_emotion']:
                    validation_stats['skipped_files'] += 1
                    continue
                
                # Validate audio file
                audio_path = os.path.join(actor_path, audio_file)
                if not validate_audio_file(audio_path, FEATURE_CONFIG['duration'], FEATURE_CONFIG['sample_rate']):
                    validation_stats['invalid_files'] += 1
                    continue
                
                # Process valid audio
                audio = librosa.load(audio_path, sr=FEATURE_CONFIG['sample_rate'])[0]
                target_length = FEATURE_CONFIG['sample_rate'] * FEATURE_CONFIG['duration']
                if len(audio) < target_length:
                    audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
                else:
                    audio = audio[:target_length]
                
                # Original audio
                feature_vector = extract_features(audio, sample_rate=FEATURE_CONFIG['sample_rate'])
                features.append(feature_vector)
                labels.append(emotion_id - 1)
                emotion_counts[emotion] += 1
                validation_stats['valid_files'] += 1
                
                # Augmented audio
                for _ in range(DATASET_CONFIG['augmentations_per_sample']):
                    augmented_audio = augment_audio(audio, sample_rate=FEATURE_CONFIG['sample_rate'])
                    if not validate_audio(augmented_audio, audio_path + " (augmented)", FEATURE_CONFIG['sample_rate'], FEATURE_CONFIG['duration']):
                        continue
                    feature_vector = extract_features(augmented_audio, sample_rate=FEATURE_CONFIG['sample_rate'])
                    features.append(feature_vector)
                    labels.append(emotion_id - 1)
                    validation_stats['valid_files'] += 1
            except Exception as e:
                logging.error(f"Error processing {audio_file}: {str(e)}")
                validation_stats['invalid_files'] += 1
    
    # Process EMODB dataset
    logging.info("Processing EMODB dataset...")
    for audio_file in tqdm(os.listdir(emodb_dir), desc="Processing EMODB"):
        validation_stats['total_files'] += 1
        if not audio_file.endswith('.wav'):
            validation_stats['skipped_files'] += 1
            continue
        
        try:
            # Extract and validate emotion code
            if len(audio_file) < 6:
                logging.warning(f"Invalid EMODB filename format: {audio_file}")
                validation_stats['invalid_files'] += 1
                continue
            emotion_code = audio_file[5]
            if not validate_emotion_label(emotion_code, 'EMODB'):
                validation_stats['invalid_files'] += 1
                continue
            
            emotion = EMODB_EMOTION_MAP[emotion_code]
            if emotion_counts[emotion] >= DATASET_CONFIG['samples_per_emotion']:
                validation_stats['skipped_files'] += 1
                continue
            
            # Validate audio file
            audio_path = os.path.join(emodb_dir, audio_file)
            if not validate_audio_file(audio_path, FEATURE_CONFIG['duration'], FEATURE_CONFIG['sample_rate']):
                validation_stats['invalid_files'] += 1
                continue
            
            # Process valid audio
            audio = librosa.load(audio_path, sr=FEATURE_CONFIG['sample_rate'])[0]
            target_length = FEATURE_CONFIG['sample_rate'] * FEATURE_CONFIG['duration']
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
            else:
                audio = audio[:target_length]
            
            # Original audio
            feature_vector = extract_features(audio, sample_rate=FEATURE_CONFIG['sample_rate'])
            features.append(feature_vector)
            labels.append(EMOTIONS.index(emotion))
            emotion_counts[emotion] += 1
            validation_stats['valid_files'] += 1
            
            # Augmented audio
            for _ in range(DATASET_CONFIG['augmentations_per_sample']):
                augmented_audio = augment_audio(audio, sample_rate=FEATURE_CONFIG['sample_rate'])
                if not validate_audio(augmented_audio, audio_path + " (augmented)", FEATURE_CONFIG['sample_rate'], FEATURE_CONFIG['duration']):
                    continue
                feature_vector = extract_features(augmented_audio, sample_rate=FEATURE_CONFIG['sample_rate'])
                features.append(feature_vector)
                labels.append(EMOTIONS.index(emotion))
                validation_stats['valid_files'] += 1
        except Exception as e:
            logging.error(f"Error processing {audio_file}: {str(e)}")
            validation_stats['invalid_files'] += 1
    
    # Log and print validation statistics
    logging.info("Validation Statistics:")
    logging.info(f"Total files processed: {validation_stats['total_files']}")
    logging.info(f"Valid files: {validation_stats['valid_files']}")
    logging.info(f"Invalid files: {validation_stats['invalid_files']}")
    logging.info(f"Skipped files: {validation_stats['skipped_files']}")
    print("Validation Statistics:")
    print(f"Total files processed: {validation_stats['total_files']}")
    print(f"Valid files: {validation_stats['valid_files']}")
    print(f"Invalid files: {validation_stats['invalid_files']}")
    print(f"Skipped files: {validation_stats['skipped_files']}")
    
    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    
    # Save preprocessed data with version
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    features_path = f'data/features_v{version}.npy'
    labels_path = f'data/labels_v{version}.npy'
    np.save(features_path, features)
    np.save(labels_path, labels)
    
    logging.info(f"Preprocessed data saved. Features shape: {features.shape}, Labels shape: {labels.shape}")
    logging.info(f"Data version: {version}")
    print(f"Preprocessed data saved as {features_path} and {labels_path}")

if __name__ == "__main__":
    ravdess_dir = "data/RAVDESS"
    emodb_dir = "data/EMODB"
    
    if not os.path.exists(ravdess_dir) or not os.path.exists(emodb_dir):
        print("Please download the RAVDESS and EMODB datasets:")
        print("RAVDESS: https://zenodo.org/record/1188976")
        print("EMODB: Search for 'Berlin Database of Emotional Speech'")
    else:
        preprocess_dataset(ravdess_dir, emodb_dir)