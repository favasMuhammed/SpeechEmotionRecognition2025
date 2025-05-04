"""
Test suite for the Streamlit app functionality.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os
from src.utils.audio_utils import record_audio, preprocess_audio
from src.utils.model_config import EMOTIONS
from src.utils.logging_utils import setup_logger

# Set up logger
logger = setup_logger('test_app', 'test_app')

class TestApp(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        os.makedirs("models", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        # Mock preprocessed data
        self.features = np.random.randn(10, 72)
        self.labels = np.random.randint(0, len(EMOTIONS), 10)
        np.save("data/features_v20250504_120000.npy", self.features)
        np.save("data/labels_v20250504_120000.npy", self.labels)
        
        # Mock model
        self.model_path = "models/ser_model_v20250504_120000.h5"
        self.model = MagicMock()
        self.model.predict.return_value = np.random.rand(1, len(EMOTIONS))

    @patch('streamlit.button')
    @patch('streamlit.selectbox')
    @patch('streamlit.write')
    @patch('streamlit.pyplot')
    @patch('streamlit.table')
    @patch('tensorflow.keras.models.load_model')
    @patch('src.utils.audio_utils.record_audio')
    def test_app(self, mock_record_audio, mock_load_model, mock_table, mock_pyplot, mock_write, mock_selectbox, mock_button):
        """Test the Streamlit app functionality."""
        from src.app import main
        
        # Mock Streamlit components
        mock_button.side_effect = [True, False]  # Simulate button click
        mock_selectbox.side_effect = [
            "features_v20250504_120000.npy",  # Data version
            "ser_model_v20250504_120000.h5"   # Model version
        ]
        mock_record_audio.return_value = np.zeros(44100 * 5)  # 5 seconds of audio
        mock_load_model.return_value = self.model
        
        # Run the app
        main()
        
        # Verify interactions
        mock_record_audio.assert_called_once()
        self.model.predict.assert_called_once()
        mock_write.assert_called()
        mock_pyplot.assert_called()
        mock_table.assert_called()

    def test_audio_recording(self):
        """Test audio recording functionality."""
        with patch('sounddevice.rec') as mock_rec:
            mock_rec.return_value = np.zeros(44100 * 5)  # 5 seconds of audio
            
            audio = record_audio(duration=5, sample_rate=44100)
            
            self.assertEqual(len(audio), 44100 * 5, "Audio length mismatch")
            self.assertEqual(audio.dtype, np.float32, "Audio dtype mismatch")

    def test_audio_preprocessing(self):
        """Test audio preprocessing functionality."""
        # Create test audio
        audio = np.random.randn(44100 * 5)  # 5 seconds of audio
        
        # Preprocess audio
        features = preprocess_audio(audio, sample_rate=44100)
        
        # Check feature shape
        self.assertEqual(features.shape[1], 72, "Feature shape mismatch")  # 20 MFCCs + 12 chroma + 40 mel

    def test_model_loading(self):
        """Test model loading functionality."""
        with patch('tensorflow.keras.models.load_model') as mock_load_model:
            mock_load_model.return_value = self.model
            
            from src.app import load_model_and_data
            features, labels, model = load_model_and_data(
                "features_v20250504_120000.npy",
                "ser_model_v20250504_120000.h5"
            )
            
            self.assertIsNotNone(features, "Features not loaded")
            self.assertIsNotNone(labels, "Labels not loaded")
            self.assertIsNotNone(model, "Model not loaded")

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        if os.path.exists("models"):
            shutil.rmtree("models")
        if os.path.exists("data"):
            shutil.rmtree("data")
        if os.path.exists("logs"):
            shutil.rmtree("logs")

if __name__ == '__main__':
    unittest.main() 