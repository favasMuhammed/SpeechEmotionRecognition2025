import unittest
import numpy as np
import soundfile as sf
import os
import librosa
from src.utils.audio_utils import (
    record_audio,
    augment_audio,
    extract_features,
    load_audio_file,
    preprocess_audio
)
from src.utils.model_config import FEATURE_CONFIG

class TestAudioUtils(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.sample_rate = FEATURE_CONFIG['sample_rate']
        self.duration = FEATURE_CONFIG['duration']
        
        # Create test directory
        if not os.path.exists('tests/data'):
            os.makedirs('tests/data')
        
        # Generate test audio with multiple frequencies
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        self.test_audio = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
        
        # Save test audio
        self.test_audio_path = 'tests/data/test_audio.wav'
        sf.write(self.test_audio_path, self.test_audio, self.sample_rate)
        
        # Create a silent audio file for testing
        self.silent_audio = np.zeros(int(self.sample_rate * self.duration))
        self.silent_audio_path = 'tests/data/silent_audio.wav'
        sf.write(self.silent_audio_path, self.silent_audio, self.sample_rate)
        
        # Create a noisy audio file for testing
        self.noisy_audio = self.test_audio + 0.1 * np.random.randn(len(self.test_audio))
        self.noisy_audio_path = 'tests/data/noisy_audio.wav'
        sf.write(self.noisy_audio_path, self.noisy_audio, self.sample_rate)
    
    def test_record_audio(self):
        """Test audio recording functionality."""
        # Test normal recording
        audio = record_audio(duration=1, sample_rate=self.sample_rate)
        self.assertEqual(len(audio), self.sample_rate, "Recorded audio length mismatch.")
        self.assertTrue(np.all(np.isfinite(audio)), "Recorded audio contains invalid values.")
        
        # Test recording with different duration
        audio = record_audio(duration=2, sample_rate=self.sample_rate)
        self.assertEqual(len(audio), 2 * self.sample_rate, "Recorded audio length mismatch.")
        
        # Test recording with different sample rate
        audio = record_audio(duration=1, sample_rate=8000)
        self.assertEqual(len(audio), 8000, "Recorded audio length mismatch.")
    
    def test_augment_audio(self):
        """Test audio augmentation."""
        # Test normal augmentation
        augmented_audio = augment_audio(self.test_audio, sample_rate=self.sample_rate)
        self.assertEqual(len(augmented_audio), len(self.test_audio), "Augmented audio length mismatch.")
        self.assertTrue(np.all(np.isfinite(augmented_audio)), "Augmented audio contains invalid values.")
        self.assertFalse(np.array_equal(augmented_audio, self.test_audio), "Augmented audio is identical to original.")
        
        # Test augmentation of silent audio
        augmented_silent = augment_audio(self.silent_audio, sample_rate=self.sample_rate)
        self.assertEqual(len(augmented_silent), len(self.silent_audio), "Augmented silent audio length mismatch.")
        
        # Test augmentation of noisy audio
        augmented_noisy = augment_audio(self.noisy_audio, sample_rate=self.sample_rate)
        self.assertEqual(len(augmented_noisy), len(self.noisy_audio), "Augmented noisy audio length mismatch.")
        
        # Test augmentation preserves signal characteristics
        original_energy = np.sum(self.test_audio ** 2)
        augmented_energy = np.sum(augmented_audio ** 2)
        self.assertAlmostEqual(original_energy, augmented_energy, delta=original_energy * 0.5,
                             msg="Augmentation significantly changed signal energy.")
    
    def test_extract_features(self):
        """Test feature extraction."""
        # Test normal feature extraction
        features = extract_features(self.test_audio, sample_rate=self.sample_rate)
        expected_features = (1, 72)  # 20 MFCCs + 12 chroma + 40 mel
        self.assertEqual(features.shape, expected_features, "Feature vector shape mismatch.")
        self.assertTrue(np.all(np.isfinite(features)), "Features contain invalid values.")
        
        # Test feature extraction from silent audio
        silent_features = extract_features(self.silent_audio, sample_rate=self.sample_rate)
        self.assertEqual(silent_features.shape, expected_features, "Silent audio feature shape mismatch.")
        
        # Test feature extraction from noisy audio
        noisy_features = extract_features(self.noisy_audio, sample_rate=self.sample_rate)
        self.assertEqual(noisy_features.shape, expected_features, "Noisy audio feature shape mismatch.")
        
        # Test feature values are within expected ranges
        self.assertTrue(np.all(features >= -100), "Features contain values below -100.")
        self.assertTrue(np.all(features <= 100), "Features contain values above 100.")
    
    def test_load_audio_file(self):
        """Test audio file loading."""
        # Test normal audio loading
        audio, sr = load_audio_file(self.test_audio_path, target_sr=self.sample_rate)
        self.assertEqual(len(audio), len(self.test_audio), "Loaded audio length mismatch.")
        self.assertTrue(np.all(np.isfinite(audio)), "Loaded audio contains invalid values.")
        self.assertEqual(sr, self.sample_rate, "Sample rate mismatch.")
        
        # Test loading with different sample rate
        audio, sr = load_audio_file(self.test_audio_path, target_sr=8000)
        self.assertEqual(len(audio), int(8000 * self.duration), "Resampled audio length mismatch.")
        self.assertEqual(sr, 8000, "Resampled sample rate mismatch.")
        
        # Test loading non-existent file
        with self.assertRaises(FileNotFoundError):
            load_audio_file('nonexistent.wav', target_sr=self.sample_rate)
    
    def test_preprocess_audio(self):
        """Test audio preprocessing."""
        # Test normal preprocessing
        features = preprocess_audio(self.test_audio, sample_rate=self.sample_rate)
        expected_features = (1, 72)  # 20 MFCCs + 12 chroma + 40 mel
        self.assertEqual(features.shape, expected_features, "Preprocessed feature vector shape mismatch.")
        self.assertTrue(np.all(np.isfinite(features)), "Preprocessed features contain invalid values.")
        
        # Test preprocessing of silent audio
        silent_features = preprocess_audio(self.silent_audio, sample_rate=self.sample_rate)
        self.assertEqual(silent_features.shape, expected_features, "Silent audio preprocessed feature shape mismatch.")
        
        # Test preprocessing of noisy audio
        noisy_features = preprocess_audio(self.noisy_audio, sample_rate=self.sample_rate)
        self.assertEqual(noisy_features.shape, expected_features, "Noisy audio preprocessed feature shape mismatch.")
        
        # Test preprocessing with different sample rate
        features = preprocess_audio(self.test_audio, sample_rate=8000)
        self.assertEqual(features.shape, expected_features, "Resampled audio preprocessed feature shape mismatch.")
    
    def tearDown(self):
        """Clean up test files."""
        for file_path in [self.test_audio_path, self.silent_audio_path, self.noisy_audio_path]:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists('tests/data'):
            os.rmdir('tests/data')

if __name__ == '__main__':
    unittest.main() 