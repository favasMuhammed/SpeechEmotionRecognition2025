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

class TestPreprocessData(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.sample_rate = FEATURE_CONFIG['sample_rate']
        self.duration = FEATURE_CONFIG['duration']
        
        # Create test directories
        os.makedirs("tests/data", exist_ok=True)
        os.makedirs("data/RAVDESS", exist_ok=True)
        os.makedirs("data/EMODB", exist_ok=True)
        
        # Check if datasets are available
        self.datasets_available = {
            'ravdess': os.path.exists(DATASET_CONFIG['ravdess_path']) and os.listdir(DATASET_CONFIG['ravdess_path']),
            'emodb': os.path.exists(DATASET_CONFIG['emodb_path']) and os.listdir(DATASET_CONFIG['emodb_path'])
        }
        
        # Create test files if datasets are not available
        if not self.datasets_available['ravdess']:
            self.test_files['ravdess_valid'] = os.path.join("data/RAVDESS", "mock_ravdess.wav")
            sf.write(self.test_files['ravdess_valid'], np.zeros(44100 * 5), 44100)
        
        if not self.datasets_available['emodb']:
            self.test_files['emodb_valid'] = os.path.join("data/EMODB", "mock_emodb.wav")
            sf.write(self.test_files['emodb_valid'], np.zeros(44100 * 5), 44100)
    
    def test_validate_audio_file(self):
        """Test audio file validation."""
        from preprocess_data import validate_audio_file
        
        # Test valid audio file
        is_valid, message = validate_audio_file(self.test_files['ravdess_valid'])
        self.assertTrue(is_valid, f"Valid audio file failed validation: {message}")
        
        # Test silent audio file
        is_valid, message = validate_audio_file(self.test_files['ravdess_silent'])
        self.assertFalse(is_valid, "Silent audio file passed validation")
        self.assertIn("silent", message.lower(), "Incorrect validation message for silent audio")
        
        # Test short audio file
        is_valid, message = validate_audio_file(self.test_files['ravdess_short'])
        self.assertFalse(is_valid, "Short audio file passed validation")
        self.assertIn("duration", message.lower(), "Incorrect validation message for short audio")
        
        # Test non-existent file
        is_valid, message = validate_audio_file('nonexistent.wav')
        self.assertFalse(is_valid, "Non-existent file passed validation")
        self.assertIn("not found", message.lower(), "Incorrect validation message for non-existent file")
    
    def test_validate_emotion_label(self):
        """Test emotion label validation."""
        from preprocess_data import validate_emotion_label
        
        # Test valid RAVDESS labels
        for emotion in EMOTIONS:
            if emotion != 'boredom':  # RAVDESS doesn't have boredom
                is_valid, message = validate_emotion_label(emotion, 'ravdess')
                self.assertTrue(is_valid, f"Valid RAVDESS label failed validation: {message}")
        
        # Test valid EMODB labels
        for code in EMODB_EMOTION_MAP.keys():
            is_valid, message = validate_emotion_label(code, 'emodb')
            self.assertTrue(is_valid, f"Valid EMODB code {code} failed validation: {message}")
        
        # Test invalid labels
        is_valid, message = validate_emotion_label('invalid_emotion', 'ravdess')
        self.assertFalse(is_valid, "Invalid label passed validation")
        self.assertIn("invalid", message.lower(), "Incorrect validation message for invalid label")
    
    def test_process_ravdess_audio(self):
        """Test RAVDESS audio processing."""
        if not self.datasets_available['ravdess']:
            self.skipTest("RAVDESS dataset not available")
        
        from preprocess_data import process_ravdess_audio
        features, label = process_ravdess_audio(self.test_files['ravdess_valid'])
        
        self.assertIsNotNone(features, "Features should not be None")
        self.assertIsNotNone(label, "Label should not be None")
        self.assertEqual(features.shape[1], 72, "Feature shape mismatch")
    
    def test_process_emodb_audio(self):
        """Test EMODB audio processing."""
        if not self.datasets_available['emodb']:
            self.skipTest("EMODB dataset not available")
        
        from preprocess_data import process_emodb_audio
        features, label = process_emodb_audio(self.test_files['emodb_valid'])
        
        self.assertIsNotNone(features, "Features should not be None")
        self.assertIsNotNone(label, "Label should not be None")
        self.assertEqual(features.shape[1], 72, "Feature shape mismatch")
    
    def test_preprocess_dataset(self):
        """Test dataset preprocessing."""
        if not any(self.datasets_available.values()):
            self.skipTest("No datasets available for testing")
        
        from preprocess_data import preprocess_dataset
        features, labels = preprocess_dataset()
        
        self.assertIsNotNone(features, "Features should not be None")
        self.assertIsNotNone(labels, "Labels should not be None")
        self.assertEqual(features.shape[0], labels.shape[0], "Number of features and labels should match")
    
    def tearDown(self):
        """Clean up test files and directories."""
        for file_path in self.test_files.values():
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists('tests/data'):
            os.rmdir('tests/data')

if __name__ == '__main__':
    unittest.main() 