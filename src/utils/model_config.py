"""
Configuration settings for the Speech Emotion Recognition model.
"""

import os
from typing import Dict
import numpy as np

# Configuration version
CONFIG_VERSION = "1.1.0"

# Dataset paths
DATASET_PATHS = {
    'RAVDESS': os.path.join('data', 'RAVDESS'),
    'EMODB': os.path.join('data', 'EMODB'),
    'processed': os.path.join('data', 'processed')
}

# Supported emotions
EMOTIONS = [
    'angry', 'boredom', 'calm', 'disgust', 'fear',
    'happy', 'neutral', 'sad', 'surprise'
]

# Emotion mappings for different datasets
EMODB_EMOTION_MAP = {
    'W': 'angry',    # Wut
    'L': 'boredom',  # Langeweile
    'E': 'disgust',  # Ekel
    'A': 'fear',     # Angst
    'F': 'happy',    # Freude
    'T': 'sad',      # Trauer
    'N': 'neutral'   # Neutral
}

RAVDESS_EMOTION_MAP = {
    1: 'neutral',
    2: 'calm',
    3: 'happy',
    4: 'sad',
    5: 'angry',
    6: 'fear',
    7: 'disgust',
    8: 'surprise'
}

# Combine emotion mappings
EMOTION_MAPPINGS = {
    'EMODB': EMODB_EMOTION_MAP,
    'RAVDESS': RAVDESS_EMOTION_MAP
}

# Model configuration
MODEL_CONFIG = {
    'version': CONFIG_VERSION,
    'model': {
        'input_shape': (72,),
        'conv_filters': [128, 256, 512],  # Increased filter sizes
        'conv_kernel_size': 3,
        'lstm_units': [256, 128],  # Increased LSTM units
        'dense_units': [512, 256, 128],  # Increased dense units
        'dropout_rate': 0.5,  # Increased dropout
        'l2_reg': 5e-5,  # Reduced L2 regularization
        'batch_norm': True,
        'attention_heads': 4,  # Multi-head attention
        'residual_connections': True,  # Enable residual connections
        'se_ratio': 16  # Squeeze-and-Excitation ratio
    },
    'training': {
        'batch_size': 32,  # Reduced batch size for better generalization
        'epochs': 200,  # Increased epochs
        'early_stopping_patience': 20,  # Increased patience
        'learning_rate': 0.0005,  # Reduced learning rate
        'reduce_lr_factor': 0.2,  # More aggressive LR reduction
        'reduce_lr_patience': 8,  # Increased patience
        'reduce_lr_min_lr': 1e-7,  # Lower minimum LR
        'warmup_epochs': 10,  # Increased warmup
        'class_weights': True,
        'focal_loss': {
            'gamma': 2.5,  # Increased gamma
            'alpha': 0.3  # Adjusted alpha
        },
        'mixup_alpha': 0.2,  # Added mixup augmentation
        'label_smoothing': 0.1,  # Added label smoothing
        'gradient_clip_norm': 1.0  # Added gradient clipping
    },
    'feature_extraction': {
        'normalize_features': True,
        'feature_selection': True,  # Enable feature selection
        'n_features': 72,
        'augmentation': {
            'time_warp': True,
            'pitch_shift': True,
            'noise': True,
            'spec_augment': True
        }
    }
}

# Feature extraction configuration
FEATURE_CONFIG = {
    'n_mfcc': 40,
    'n_mels': 8,
    'n_chroma': 24,
    'n_contrast': 6,
    'n_tonnetz': 6,
    'hop_length': 512,
    'n_fft': 2048,
    'n_features': 72,  # Updated to match our feature vector size
    'sample_rate': 22050,
    'duration': 3.0,
    'delta_features': True,
    'delta2_features': True
}

# Audio processing configuration
AUDIO_CONFIG = {
    'min_duration': 1.0,  # Minimum duration in seconds
    'max_duration': 5.5,  # Maximum duration in seconds
    'min_amplitude': 0.001,  # Minimum amplitude threshold
    'sample_rate': 16000,  # Target sample rate
    'n_fft': 2048,  # FFT window size
    'hop_length': 512,  # Number of samples between FFT windows
    'n_mels': 32,  # Number of mel bands
    'fmin': 50,  # Minimum frequency
    'fmax': 8000,  # Maximum frequency
    'duration': 3.0  # Default recording duration in seconds
}

def validate_config() -> bool:
    """
    Validate configuration settings.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    try:
        # Validate model architecture
        if not all(isinstance(x, int) and x > 0 for x in MODEL_CONFIG['model']['conv_filters']):
            raise ValueError("CNN filters must be positive integers")
        if not isinstance(MODEL_CONFIG['model']['conv_kernel_size'], int) or MODEL_CONFIG['model']['conv_kernel_size'] <= 0:
            raise ValueError("CNN kernel size must be a positive integer")
        if not all(isinstance(x, int) and x > 0 for x in MODEL_CONFIG['model']['lstm_units']):
            raise ValueError("LSTM units must be positive integers")
        if not all(isinstance(x, int) and x > 0 for x in MODEL_CONFIG['model']['dense_units']):
            raise ValueError("Dense units must be positive integers")
        if not 0 <= MODEL_CONFIG['model']['dropout_rate'] <= 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        
        # Validate training parameters
        if not isinstance(MODEL_CONFIG['training']['batch_size'], int) or MODEL_CONFIG['training']['batch_size'] <= 0:
            raise ValueError("Batch size must be a positive integer")
        if not isinstance(MODEL_CONFIG['training']['epochs'], int) or MODEL_CONFIG['training']['epochs'] <= 0:
            raise ValueError("Number of epochs must be a positive integer")
        if not isinstance(MODEL_CONFIG['training']['early_stopping_patience'], int) or MODEL_CONFIG['training']['early_stopping_patience'] <= 0:
            raise ValueError("Early stopping patience must be a positive integer")
        if not isinstance(MODEL_CONFIG['training']['learning_rate'], float) or MODEL_CONFIG['training']['learning_rate'] <= 0:
            raise ValueError("Learning rate must be a positive float")
        
        # Validate feature extraction parameters
        if not all(isinstance(x, int) and x > 0 for x in [
            FEATURE_CONFIG['n_mfcc'],
            FEATURE_CONFIG['n_mels'],
            FEATURE_CONFIG['n_chroma'],
            FEATURE_CONFIG['n_contrast'],
            FEATURE_CONFIG['n_tonnetz'],
            FEATURE_CONFIG['hop_length'],
            FEATURE_CONFIG['n_fft']
        ]):
            raise ValueError("Feature extraction parameters must be positive integers")
        
        # Validate audio processing parameters
        if not isinstance(AUDIO_CONFIG['sample_rate'], int) or AUDIO_CONFIG['sample_rate'] <= 0:
            raise ValueError("Sample rate must be a positive integer")
        if not isinstance(AUDIO_CONFIG['min_duration'], float) or AUDIO_CONFIG['min_duration'] <= 0:
            raise ValueError("Minimum duration must be a positive float")
        if not isinstance(AUDIO_CONFIG['max_duration'], float) or AUDIO_CONFIG['max_duration'] <= AUDIO_CONFIG['min_duration']:
            raise ValueError("Maximum duration must be greater than minimum duration")
    
    # Validate dataset paths
        for path in DATASET_PATHS.values():
            if not isinstance(path, str):
                raise ValueError("Dataset paths must be strings")
        
        return True
        
    except Exception as e:
        print(f"Configuration validation error: {str(e)}")
        return False