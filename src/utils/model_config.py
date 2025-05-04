"""
Configuration settings for the Speech Emotion Recognition model.
"""

# List of emotions supported by the model
EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'ps', 'boredom']

# EMODB emotion codes mapping (official documentation)
EMODB_EMOTION_MAP = {
    'W': 'angry',      # Ärger (Wut) - Anger
    'L': 'boredom',    # Langeweile - Boredom
    'E': 'disgust',    # Ekel - Disgust
    'A': 'fear',       # Angst - Anxiety/Fear
    'F': 'happy',      # Freude - Happiness
    'T': 'sad',        # Trauer - Sadness
    'N': 'neutral'     # Neutral
}

# Model architecture configuration
MODEL_CONFIG = {
    'conv_layers': 3,
    'conv_filters': [64, 128, 256],
    'conv_kernel_size': 3,
    'conv_strides': 1,
    'pool_size': 2,
    'lstm_units': [128, 64],
    'dense_units': 64,
    'dropout_rate': 0.3
}

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'lr_decay_steps': 1000,
    'lr_decay_rate': 0.9,
    'validation_split': 0.2
}

# Feature extraction configuration
FEATURE_CONFIG = {
    'sample_rate': 44100,
    'duration': 5,  # seconds
    'n_mfcc': 20,
    'n_chroma': 12,
    'n_mel': 40,
    'hop_length': 512,
    'n_fft': 2048
}

# Dataset configuration
DATASET_CONFIG = {
    'ravdess_path': 'data/RAVDESS',
    'emodb_path': 'data/EMODB',
    'test_size': 0.2,
    'random_state': 42
} 