"""
Test suite for the model training functionality.
"""

import unittest
import numpy as np
import os
import tensorflow as tf
from src.utils.model_config import MODEL_CONFIG, EMOTIONS, DATASET_CONFIG, TRAINING_CONFIG
from src.utils.logging_utils import setup_logger

# Set up logger
logger = setup_logger('test_train_model', 'test_train')

class TestTrainModel(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.input_shape = (72,)  # Feature vector size: 20 MFCCs + 12 chroma + 40 mel
        self.X_train = np.random.randn(10, 72)  # 10 samples
        self.y_train = np.random.randint(0, len(EMOTIONS), 10)
        self.X_test = np.random.randn(5, 72)  # 5 samples
        self.y_test = np.random.randint(0, len(EMOTIONS), 5)
        
        # Create directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        # Save mock preprocessed data
        np.save("data/features_v20250504_120000.npy", np.vstack((self.X_train, self.X_test)))
        np.save("data/labels_v20250504_120000.npy", np.concatenate((self.y_train, self.y_test)))

    def test_create_model(self):
        """Test model creation."""
        from src.train_model import create_model
        
        model = create_model(self.input_shape)
        self.assertIsInstance(model, tf.keras.Model, "Model is not a tf.keras.Model instance.")
        self.assertEqual(model.input_shape, (None, 72, 1), "Input shape mismatch.")
        self.assertEqual(model.output_shape, (None, len(EMOTIONS)), "Output shape mismatch.")
        
        # Check layer types and counts
        conv_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv1D)]
        lstm_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.LSTM)]
        dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
        
        self.assertEqual(len(conv_layers), 3, "Expected 3 Conv1D layers")
        self.assertEqual(len(lstm_layers), 2, "Expected 2 LSTM layers")
        self.assertEqual(len(dense_layers), 2, "Expected 2 Dense layers")

    def test_model_training(self):
        """Test model training process."""
        from src.train_model import create_model
        
        # Create and compile model
        model = create_model(self.input_shape)
        
        # Mock training process
        history = model.fit(
            self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1),
            self.y_train,
            epochs=1,
            batch_size=2,
            validation_data=(
                self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1),
                self.y_test
            )
        )
        
        # Check training history
        self.assertIn('loss', history.history, "Training history missing loss")
        self.assertIn('accuracy', history.history, "Training history missing accuracy")
        self.assertIn('val_loss', history.history, "Training history missing validation loss")
        self.assertIn('val_accuracy', history.history, "Training history missing validation accuracy")

    def test_model_evaluation(self):
        """Test model evaluation."""
        from src.train_model import create_model
        
        # Create and compile model
        model = create_model(self.input_shape)
        
        # Train for one epoch
        model.fit(
            self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1),
            self.y_train,
            epochs=1,
            batch_size=2
        )
        
        # Evaluate model
        loss, accuracy = model.evaluate(
            self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1),
            self.y_test
        )
        
        # Check evaluation metrics
        self.assertIsInstance(loss, float, "Loss should be a float")
        self.assertIsInstance(accuracy, float, "Accuracy should be a float")
        self.assertGreaterEqual(accuracy, 0.0, "Accuracy should be non-negative")
        self.assertLessEqual(accuracy, 1.0, "Accuracy should be less than or equal to 1.0")

    def test_model_export(self):
        """Test model export functionality."""
        from src.train_model import create_model, export_model
        
        # Create and compile model
        model = create_model(self.input_shape)
        
        # Export model
        version = "test_version"
        export_dir = "models"
        export_model(model, version, export_dir)
        
        # Check exported files
        self.assertTrue(os.path.exists(f"{export_dir}/ser_model_v{version}.h5"),
                       "H5 model file not created")
        self.assertTrue(os.path.exists(f"{export_dir}/ser_model_v{version}_savedmodel"),
                       "SavedModel directory not created")
        self.assertTrue(os.path.exists(f"{export_dir}/ser_model_v{version}.onnx"),
                       "ONNX model file not created")

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