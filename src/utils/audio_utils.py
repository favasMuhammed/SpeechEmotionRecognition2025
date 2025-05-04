"""
Audio processing utilities for the Speech Emotion Recognition system.
"""

import numpy as np
import librosa
import sounddevice as sd
from src.utils.logging_utils import setup_logger
from src.utils.model_config import FEATURE_CONFIG

# Set up logger
logger = setup_logger('audio_utils', 'audio')

def augment_audio(audio, sample_rate, pitch_factor=0.2, speed_factor=0.2, noise_factor=0.005):
    """
    Apply augmentation to an audio signal.

    Args:
        audio (numpy.ndarray): Input audio signal.
        sample_rate (int): Sample rate of the audio.
        pitch_factor (float): Maximum pitch shift in semitones.
        speed_factor (float): Maximum speed change factor.
        noise_factor (float): Amplitude of added noise.

    Returns:
        numpy.ndarray: Augmented audio signal.
    """
    try:
        # Pitch shifting
        pitch_shift = np.random.uniform(-pitch_factor, pitch_factor) * 2
        audio_shifted = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_shift)
        logger.debug(f"Applied pitch shift: {pitch_shift:.2f} semitones")

        # Time stretching (speed change)
        speed_change = np.random.uniform(1 - speed_factor, 1 + speed_factor)
        audio_stretched = librosa.effects.time_stretch(audio_shifted, rate=speed_change)
        logger.debug(f"Applied time stretch: {speed_change:.2f}x")

        # Add random noise
        noise = np.random.normal(0, noise_factor, len(audio_stretched))
        audio_noisy = audio_stretched + noise
        logger.debug(f"Added noise with factor: {noise_factor}")

        # Ensure the audio length matches the original
        target_length = len(audio)
        if len(audio_noisy) < target_length:
            audio_noisy = np.pad(audio_noisy, (0, target_length - len(audio_noisy)), 'constant')
        else:
            audio_noisy = audio_noisy[:target_length]

        return audio_noisy
    except Exception as e:
        logger.error(f"Error in audio augmentation: {str(e)}")
        raise

def record_audio(duration=5, sample_rate=44100):
    """
    Record audio from the microphone.

    Args:
        duration (int): Duration of recording in seconds.
        sample_rate (int): Sample rate of the recording.

    Returns:
        numpy.ndarray: Recorded audio signal.
    """
    try:
        logger.info(f"Recording audio for {duration} seconds...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
        sd.wait()
        logger.info("Recording completed")
        return audio.flatten().astype(np.float32)  # Ensure float32 dtype
    except Exception as e:
        logger.error(f"Error recording audio: {str(e)}")
        raise

def load_audio_file(file_path, target_sr=None):
    """
    Load an audio file and optionally resample it.

    Args:
        file_path (str): Path to the audio file.
        target_sr (int, optional): Target sample rate for resampling.

    Returns:
        tuple: (audio signal, sample rate)
    """
    try:
        logger.debug(f"Loading audio file: {file_path}")
        audio, sr = librosa.load(file_path, sr=target_sr)
        logger.debug(f"Loaded audio with sample rate: {sr}")
        return audio, sr
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {str(e)}")
        raise

def extract_features(audio, sample_rate):
    """
    Extract audio features (MFCCs, chroma, mel-spectrogram).

    Args:
        audio (numpy.ndarray): Audio signal.
        sample_rate (int): Sample rate of the audio.

    Returns:
        numpy.ndarray: Feature vector with shape (1, 72).
    """
    try:
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=sample_rate,
            n_mfcc=FEATURE_CONFIG['n_mfcc'],
            n_fft=FEATURE_CONFIG['n_fft'],
            hop_length=FEATURE_CONFIG['hop_length']
        )
        
        # Extract chroma features
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=sample_rate,
            n_fft=FEATURE_CONFIG['n_fft'],
            hop_length=FEATURE_CONFIG['hop_length']
        )
        
        # Extract mel-spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_mels=FEATURE_CONFIG['n_mel'],
            n_fft=FEATURE_CONFIG['n_fft'],
            hop_length=FEATURE_CONFIG['hop_length']
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        
        # Compute mean of each feature over time
        mfccs_mean = np.mean(mfccs, axis=1)
        chroma_mean = np.mean(chroma, axis=1)
        mel_mean = np.mean(mel, axis=1)
        
        # Concatenate features and reshape to (1, 72)
        features = np.concatenate([mfccs_mean, chroma_mean, mel_mean])
        features = features.reshape(1, -1)  # Reshape to (1, 72)
        
        logger.debug(f"Extracted features with shape: {features.shape}")
        return features
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        raise

def preprocess_audio(audio, sample_rate=44100):
    """
    Preprocess audio signal by normalizing and extracting features.

    Args:
        audio (numpy.ndarray): Audio signal.
        sample_rate (int): Sample rate of the audio.

    Returns:
        numpy.ndarray: Preprocessed feature vector.
    """
    try:
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        # Extract features
        features = extract_features(audio, sample_rate)
        
        # Reshape for model input (add channel dimension)
        features = features.reshape(1, -1)
        
        return features
    except Exception as e:
        logger.error(f"Error preprocessing audio: {str(e)}")
        raise 