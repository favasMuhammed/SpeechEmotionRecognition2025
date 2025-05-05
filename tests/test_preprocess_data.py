import unittest
import numpy as np
import os
import librosa
import soundfile as sf
from src.utils.audio_utils import (
    augment_audio,
    extract_features,
    load_audio_file,
    preprocess_audio
)
from src.utils.model_config import (
    FEATURE_CONFIG,
    DATASET_CONFIG,
    EMOTIONS,
    EMODB_EMOTION_MAP
)
from src.preprocess_data import (
    validate_audio,
    validate_audio_file,
    validate_emotion_label,
    preprocess_dataset
)
from datetime import datetime
import shutil

class TestPreprocessData(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.test_dir = 'tests/test_files'
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, 'wav'), exist_ok=True)
        
        # Create test audio files
        self.duration = 5.0
        self.sample_rate = 48000
        t = np.linspace(0, self.duration, int(self.duration * self.sample_rate))
        test_signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        self.test_files = {
            'ravdess_valid': os.path.join(self.test_dir, '03-01-01-01-01-01-01.wav'),
            'emodb_valid': os.path.join(self.test_dir, 'wav', '03a01Fa.wav'),
            'silent': os.path.join(self.test_dir, 'silent.wav'),
            'short': os.path.join(self.test_dir, 'short.wav'),
            'nan': os.path.join(self.test_dir, 'nan.wav')
        }
        
        # Save valid test files
        sf.write(self.test_files['ravdess_valid'], test_signal, self.sample_rate)
        sf.write(self.test_files['emodb_valid'], test_signal, self.sample_rate)
        
        # Save invalid test files
        sf.write(self.test_files['silent'], np.zeros_like(test_signal), self.sample_rate)
        sf.write(self.test_files['short'], test_signal[:int(2.0 * self.sample_rate)], self.sample_rate)
        sf.write(self.test_files['nan'], np.full_like(test_signal, np.nan), self.sample_rate)

    def test_validate_audio(self):
        """Test audio validation."""
        # Test valid audio
        valid_audio = np.sin(2 * np.pi * 440 * np.linspace(0, self.duration, int(self.sample_rate * self.duration)))
        self.assertTrue(validate_audio(valid_audio, "test.wav", self.sample_rate, self.duration))
        
        # Test silent audio
        silent_audio = np.zeros(int(self.sample_rate * self.duration))
        self.assertFalse(validate_audio(silent_audio, "silent.wav", self.sample_rate, self.duration))
        
        # Test short audio
        short_audio = np.zeros(int(self.sample_rate * self.duration * 0.4))
        self.assertFalse(validate_audio(short_audio, "short.wav", self.sample_rate, self.duration))
        
        # Test audio with NaN values
        nan_audio = np.full(int(self.sample_rate * self.duration), np.nan)
        self.assertFalse(validate_audio(nan_audio, "nan.wav", self.sample_rate, self.duration))
    
    def test_validate_audio_file(self):
        """Test audio file validation."""
        # Test valid audio file
        self.assertTrue(validate_audio_file(self.test_files['ravdess_valid'], self.duration, self.sample_rate))
        
        # Test silent audio file
        self.assertFalse(validate_audio_file(self.test_files['silent'], self.duration, self.sample_rate))
        
        # Test short audio file
        self.assertFalse(validate_audio_file(self.test_files['short'], self.duration, self.sample_rate))
        
        # Test non-existent file
        self.assertFalse(validate_audio_file('nonexistent.wav', self.duration, self.sample_rate))
    
    def test_validate_emotion_label(self):
        """Test emotion label validation."""
        # Test valid RAVDESS labels
        for emotion_id in range(1, 9):  # RAVDESS uses 1-8
            self.assertTrue(validate_emotion_label(emotion_id, 'RAVDESS'))
        
        # Test valid EMODB labels
        for code in EMODB_EMOTION_MAP.keys():
            self.assertTrue(validate_emotion_label(code, 'EMODB'))
        
        # Test invalid labels
        self.assertFalse(validate_emotion_label(0, 'RAVDESS'))  # Invalid RAVDESS ID
        self.assertFalse(validate_emotion_label(9, 'RAVDESS'))  # Invalid RAVDESS ID
        self.assertFalse(validate_emotion_label('invalid', 'EMODB'))  # Invalid EMODB code
        self.assertFalse(validate_emotion_label(1, 'INVALID_DATASET'))  # Invalid dataset
    def test_preprocess_dataset(self):
        """Test dataset preprocessing."""
        if not any(self.test_files.values()):
            self.skipTest("No test files available for testing")
        
        # Create test dataset structure
        os.makedirs(os.path.join('data', 'RAVDESS'), exist_ok=True)
        os.makedirs(os.path.join('data', 'EMODB'), exist_ok=True)
        
        # Copy test files to dataset directories
        shutil.copy(self.test_files['ravdess_valid'], os.path.join('data', 'RAVDESS'))
        shutil.copy(self.test_files['emodb_valid'], os.path.join('data', 'EMODB'))
        
        try:
            # Test preprocessing
            features, labels = preprocess_dataset('data/RAVDESS', 'data/EMODB')
            
            # Check shapes
            self.assertIsNotNone(features, "Features should not be None")
            self.assertIsNotNone(labels, "Labels should not be None")
            self.assertTrue(len(features) > 0, "Features array should not be empty")
            self.assertEqual(features.shape[1], 72, "Feature vector should have 72 dimensions")
            self.assertEqual(len(features), len(labels), "Number of features and labels should match")
            
        finally:
            # Clean up
            shutil.rmtree('data/RAVDESS', ignore_errors=True)
            shutil.rmtree('data/EMODB', ignore_errors=True)

    def tearDown(self):
        """Clean up test files."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

if __name__ == '__main__':
    unittest.main() 