"""
Model training script for the Speech Emotion Recognition system.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from src.utils.model_config import MODEL_CONFIG, EMOTIONS, TRAINING_CONFIG
from src.utils.logging_utils import setup_logger
from datetime import datetime

# Set up logger
logger = setup_logger('train_model', 'train')

def create_model(input_shape):
    """
    Create the CNN-LSTM model.

    Args:
        input_shape (tuple): Shape of input features.

    Returns:
        tf.keras.Model: Compiled model.
    """
    try:
        # Reshape input for Conv1D
        inputs = layers.Input(shape=(input_shape[0], 1))
        
        # CNN layers
        x = inputs
        for i in range(MODEL_CONFIG['conv_layers']):
            x = layers.Conv1D(
                filters=MODEL_CONFIG['conv_filters'][i],
                kernel_size=MODEL_CONFIG['conv_kernel_size'],
                strides=MODEL_CONFIG['conv_strides'],
                padding='same'
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.MaxPooling1D(pool_size=MODEL_CONFIG['pool_size'])(x)
        
        # LSTM layers
        for units in MODEL_CONFIG['lstm_units']:
            x = layers.LSTM(units, return_sequences=True)(x)
            x = layers.Dropout(MODEL_CONFIG['dropout_rate'])(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layer
        x = layers.Dense(MODEL_CONFIG['dense_units'])(x)
        x = layers.Dropout(MODEL_CONFIG['dropout_rate'])(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Output layer
        outputs = layers.Dense(len(EMOTIONS), activation='softmax')(x)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Learning rate schedule
        initial_learning_rate = TRAINING_CONFIG['learning_rate']
        decay_steps = TRAINING_CONFIG['lr_decay_steps']
        decay_rate = TRAINING_CONFIG['lr_decay_rate']
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps, decay_rate
        )
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Model created successfully")
        return model
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise

def export_model(model, version, export_dir):
    """
    Export the model in multiple formats.

    Args:
        model (tf.keras.Model): Trained model.
        version (str): Version string for the model.
        export_dir (str): Directory to save the model.
    """
    try:
        # Create export directory if it doesn't exist
        os.makedirs(export_dir, exist_ok=True)
        
        # Save H5 model
        h5_path = os.path.join(export_dir, f"ser_model_v{version}.h5")
        model.save(h5_path)
        logger.info(f"Saved H5 model to {h5_path}")
        
        # Save Keras format
        keras_path = os.path.join(export_dir, f"ser_model_v{version}.keras")
        model.save(keras_path)
        logger.info(f"Saved Keras model to {keras_path}")
        
        # Export to ONNX format
        import tf2onnx
        onnx_path = os.path.join(export_dir, f"ser_model_v{version}.onnx")
        tf2onnx.convert.from_keras(model, output_path=onnx_path)
        logger.info(f"Exported ONNX model to {onnx_path}")
    except Exception as e:
        logger.error(f"Error exporting model: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        # Load preprocessed data
        features = np.load('data/features.npy')
        labels = np.load('data/labels.npy')
        
        # Create and train model
        model = create_model(features.shape[1:])
        history = model.fit(
            features, labels,
            batch_size=TRAINING_CONFIG['batch_size'],
            epochs=TRAINING_CONFIG['epochs'],
            validation_split=TRAINING_CONFIG['validation_split']
        )
        
        # Export model
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_model(model, version, 'models')
        
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Error in training script: {str(e)}")
        raise 