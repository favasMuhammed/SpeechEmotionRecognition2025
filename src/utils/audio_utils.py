"""
Audio processing utilities for Speech Emotion Recognition.
"""

import numpy as np
import librosa
import sounddevice as sd
from sklearn.feature_selection import VarianceThreshold
from src.utils.logging_utils import setup_logger
from src.utils.model_config import FEATURE_CONFIG, AUDIO_CONFIG
import os
from typing import Tuple, Optional, Dict, Any
import logging

# Set up logger
logger = setup_logger('audio_utils', 'audio')

def augment_audio(
    audio: np.ndarray,
    sample_rate: int,
    augmentations: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Apply various augmentations to audio signal.

    Args:
        audio: Input audio signal
        sample_rate: Sample rate of the audio
        augmentations: Dictionary of augmentation parameters

    Returns:
        Augmented audio signal
    """
    if augmentations is None:
        augmentations = {
            'pitch_shift': True,
            'time_stretch': True,
            'noise': True,
            'speed_perturb': True
        }
    
    try:
        augmented = audio.copy()
        
        # Pitch shifting
        if augmentations.get('pitch_shift', False):
            n_steps = np.random.uniform(-2, 2)
            augmented = librosa.effects.pitch_shift(
                augmented,
                sr=sample_rate,
                n_steps=n_steps
            )
        
        # Time stretching
        if augmentations.get('time_stretch', False):
            rate = np.random.uniform(0.9, 1.1)
            augmented = librosa.effects.time_stretch(
                augmented,
                rate=rate
            )
        
        # Add noise
        if augmentations.get('noise', False):
            noise_level = np.random.uniform(0.001, 0.005)
            noise = np.random.normal(0, noise_level, len(augmented))
            augmented = augmented + noise
        
        # Speed perturbation
        if augmentations.get('speed_perturb', False):
            speed_factor = np.random.uniform(0.9, 1.1)
            augmented = librosa.effects.time_stretch(
                augmented,
                rate=speed_factor
            )
        
        # Ensure the augmented audio has the same length as the original
        target_length = int(round(len(audio)))
        aug_length = int(round(len(augmented)))
        if aug_length > target_length:
            logger.debug(f"Truncating augmented audio from {aug_length} to {target_length}")
            augmented = augmented[:target_length]
        elif aug_length < target_length:
            pad_width = target_length - aug_length
            if pad_width > 0:
                logger.debug(f"Padding augmented audio from {aug_length} to {target_length}")
                augmented = np.pad(augmented, (0, pad_width))
        else:
                logger.warning(f"No padding applied, pad_width={pad_width}")
        
        return augmented
        
    except Exception as e:
        logger.error(f"Error in audio augmentation: {str(e)}")
        return audio

def record_audio(duration=5, sample_rate=44100):
    """
    Record audio from the microphone.

    Args:
        duration (float): Duration of recording in seconds.
        sample_rate (int): Sample rate of the recording.

    Returns:
        numpy.ndarray: Recorded audio signal.
    """
    try:
        if duration <= 0:
            raise ValueError("Duration must be positive")
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        
        logger.info(f"Recording audio for {duration} seconds...")
        try:
            audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
            sd.wait()
        except sd.PortAudioError as e:
            raise RuntimeError(f"Audio recording device error: {str(e)}")
        logger.info("Recording completed")
        
        # Validate recording
        audio = audio.flatten().astype(np.float32)
        if not np.all(np.isfinite(audio)):
            raise ValueError("Recorded audio contains invalid values")
        if len(audio) != int(duration * sample_rate):
            raise ValueError("Recorded audio length mismatch")
        
        return audio
    except Exception as e:
        logger.error(f"Error recording audio: {str(e)}")
        raise

def load_audio_file(file_path, target_sr=FEATURE_CONFIG['sample_rate']):
    """
    Load an audio file and return the signal and sample rate.
    
    Args:
        file_path (str): Path to the audio file.
        target_sr (int, optional): Target sample rate. Defaults to FEATURE_CONFIG['sample_rate'].
    
    Returns:
        tuple: (audio_signal, sample_rate) or (None, None) if file is invalid
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Load audio file with original sample rate
        audio, sr = librosa.load(file_path, sr=None)
        logger.debug(f"Loaded audio file: {file_path}, sample_rate={sr}, length={len(audio)} samples")
        
        # Calculate and log duration
        duration = len(audio) / sr
        logger.debug(f"Audio duration: {duration:.3f} seconds")
        
        # Validate audio
        if not np.all(np.isfinite(audio)):
            logger.warning(f"Audio file {file_path} contains invalid values")
            return None, None
        
        if len(audio) < 512:  # Minimum reasonable length (e.g., ~10 ms at 48000 Hz)
            logger.warning(f"Audio file {file_path} is too short: {len(audio)} samples, {duration:.3f} seconds")
            return None, None
        
        if duration < AUDIO_CONFIG.get('min_duration', 0.5):
            logger.warning(f"Audio file {file_path} duration too short: {duration:.3f} seconds")
            return None, None
        
        # Resample if target_sr is specified and different from original
        if target_sr and sr != target_sr:
            if target_sr <= 0:
                raise ValueError("Target sample rate must be positive")
            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
            logger.debug(f"Resampled audio to {target_sr} Hz")
        
        return audio, sr
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {str(e)}")
        return None, None

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
            return False, f"Invalid duration: {duration:.3f}s (expected {AUDIO_CONFIG['min_duration']}–{AUDIO_CONFIG['max_duration']}s)"
        
        # Check amplitude
        if np.max(np.abs(audio)) < AUDIO_CONFIG.get('min_amplitude', 0.001):
            return False, f"Too quiet: max amplitude {np.max(np.abs(audio)):.6f}"
        
        return True, ""
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def extract_advanced_features(audio, sample_rate):
    """
    Extract advanced audio features for emotion recognition.
    """
    try:
        # Log audio properties
        logger.debug(f"Processing audio: length={len(audio)}, sample_rate={sample_rate}")
        
        # Ensure audio length is sufficient
        min_length = 2048  # Minimum length for robust feature extraction
        if len(audio) < min_length:
            logger.debug(f"Audio length {len(audio)} is too short for FFT analysis, padding to {min_length} samples")
            audio = np.pad(audio, (0, min_length - len(audio)), mode='reflect')
            logger.debug(f"Padded audio length: {len(audio)}")

        # Calculate adaptive n_fft size based on audio length
        n_fft = min(2048, len(audio))
        n_fft = max(512, n_fft)  # Ensure minimum n_fft to avoid empty filters
        hop_length = max(1, n_fft // 4)  # Ensure hop_length is at least 1
        
        logger.debug(f"Using n_fft={n_fft}, hop_length={hop_length}")

        # Adjust n_mels based on n_fft to prevent empty filters
        n_mels = min(32, max(8, n_fft // 64))  # Dynamic n_mels, ensuring reasonable range
        logger.debug(f"Using n_mels={n_mels}")

        # Extract MFCCs (10 coefficients)
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=10,
            n_fft=n_fft,
            hop_length=hop_length
        )
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        mfccs_features = np.concatenate([mfccs_mean, mfccs_std])

        # Extract chroma features (10 bins)
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=sample_rate,
            n_chroma=10,
            n_fft=n_fft,
            hop_length=hop_length
        )
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        chroma_features = np.concatenate([chroma_mean, chroma_std])

        # Extract mel spectrogram with optimized parameters
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            fmin=50,
            fmax=min(8000, sample_rate // 2)  # Adjust fmax to avoid empty filters
        )
        mel_spec_mean = np.mean(mel_spec, axis=1)
        mel_spec_std = np.std(mel_spec, axis=1)
        mel_features = np.concatenate([mel_spec_mean, mel_spec_std])

        # Combine all features
        features = np.concatenate([mfccs_features, chroma_features, mel_features])

        # Ensure feature vector has correct shape by padding or truncating
        target_length = 72
        if len(features) < target_length:
            logger.debug(f"Padding features from {len(features)} to {target_length}")
            features = np.pad(features, (0, target_length - len(features)))
        elif len(features) > target_length:
            logger.debug(f"Truncating features from {len(features)} to {target_length}")
            features = features[:target_length]

        return features
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        raise

def preprocess_audio(audio, sample_rate=None):
    """
    Preprocess audio for model input.
    """
    try:
        if not isinstance(audio, np.ndarray):
            raise TypeError("Input audio must be a numpy array")
        if len(audio) == 0:
            raise ValueError("Input audio is empty")
        if sample_rate is None:
            sample_rate = FEATURE_CONFIG['sample_rate']
        elif sample_rate <= 0:
            raise ValueError("Sample rate must be positive")

        # Normalize audio to prevent "too quiet" errors
        audio = librosa.util.normalize(audio)
        
        # Validate audio duration
        duration = len(audio) / sample_rate
        if not (AUDIO_CONFIG['min_duration'] <= duration <= AUDIO_CONFIG['max_duration']):
            raise ValueError(f"Audio duration {duration:.2f}s is outside required range "
                             f"({AUDIO_CONFIG['min_duration']}–{AUDIO_CONFIG['max_duration']}s)")
        
        # Validate audio amplitude (with lower threshold)
        min_amplitude = 0.001  # Lower threshold for normalized audio
        if np.mean(np.abs(audio)) < min_amplitude:
            logger.warning(f"Audio amplitude {np.mean(np.abs(audio)):.6f} is below threshold {min_amplitude}")
            # Instead of raising error, normalize the audio
            audio = librosa.util.normalize(audio)
        
        # Ensure audio is the correct length, with minimum length for feature extraction
        min_samples = 2048  # Minimum samples needed for feature extraction
        target_length = max(min_samples, int(round(sample_rate * FEATURE_CONFIG['duration'])))
        audio_length = len(audio)
        
        if audio_length < target_length:
            # Use reflection padding for better spectral properties
            logger.debug(f"Padding audio from {audio_length} to {target_length} samples using reflection")
            audio = np.pad(audio, (0, target_length - audio_length), mode='reflect')
        elif audio_length > target_length:
            # Take center portion of audio
            logger.debug(f"Truncating audio from {audio_length} to {target_length} samples from center")
            start = (audio_length - target_length) // 2
            audio = audio[start:start + target_length]
        
        # Extract features
        features = extract_advanced_features(audio, sample_rate)
        
        # Validate features
        if not np.all(np.isfinite(features)):
            raise ValueError("Preprocessed features contain invalid values")
        if features.shape != (72,):
            raise ValueError(f"Feature vector has incorrect shape: {features.shape} (expected (72,))")
        return features
    except Exception as e:
        logger.error(f"Error preprocessing audio: {str(e)}")
        raise