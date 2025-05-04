"""
Utility functions for audio augmentation in the Speech Emotion Recognition system.
"""

import numpy as np
import librosa
from src.utils.logging_utils import setup_logger

# Set up logger
logger = setup_logger('augmentation_utils', 'augment')

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