"""
Training script for the Speech Emotion Recognition model.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import glob
import json
from typing import Tuple, List, Dict

from src.utils.model_config import MODEL_CONFIG
from src.utils.logging_utils import setup_logger

# Setup logging
logger = setup_logger('train_model', 'train')

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # Convert sparse labels to one-hot
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[1])
        
        # Convert to float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Clip predictions to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Calculate focal loss
        ce = -y_true * tf.math.log(y_pred)
        pt = tf.exp(-ce)
        focal_loss = self.alpha * tf.pow(1 - pt, self.gamma) * ce
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

class Attention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1), initializer='normal')
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1), initializer='zeros')
        super().build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

def create_model(input_shape: Tuple[int, int], num_classes: int) -> models.Model:
    """Create the CNN-Bidirectional LSTM model with attention."""
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Reshape input for CNN
    x = layers.Reshape((input_shape[0], 1))(inputs)
    
    # CNN layers with residual connections
    for i, filters in enumerate(MODEL_CONFIG['model']['conv_filters']):
        # First conv block
        conv1 = layers.Conv1D(
            filters=filters,
            kernel_size=MODEL_CONFIG['model']['conv_kernel_size'],
            padding='same',
            kernel_regularizer=regularizers.l2(MODEL_CONFIG['model']['l2_reg'])
        )(x)
        bn1 = layers.BatchNormalization()(conv1)
        act1 = layers.Activation('relu')(bn1)
        
        # Second conv block
        conv2 = layers.Conv1D(
            filters=filters,
            kernel_size=MODEL_CONFIG['model']['conv_kernel_size'],
            padding='same',
            kernel_regularizer=regularizers.l2(MODEL_CONFIG['model']['l2_reg'])
        )(act1)
        bn2 = layers.BatchNormalization()(conv2)
        
        # Residual connection with projection if needed
        if x.shape[-1] != filters:
            x = layers.Conv1D(filters=filters, kernel_size=1, padding='same')(x)
        
        # Add residual connection
        x = layers.Add()([bn2, x])
        x = layers.Activation('relu')(x)
        
        # Pooling and dropout
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(MODEL_CONFIG['model']['dropout_rate'])(x)
    
    # Bidirectional LSTM layers
    for units in MODEL_CONFIG['model']['lstm_units']:
        # Main LSTM
        lstm_out = layers.Bidirectional(
            layers.LSTM(
                units=units,
                return_sequences=True,
                kernel_regularizer=regularizers.l2(MODEL_CONFIG['model']['l2_reg'])
            )
        )(x)
        
        # Project input if needed for residual connection
        if x.shape[-1] != lstm_out.shape[-1]:
            x = layers.Dense(lstm_out.shape[-1])(x)
        
        # Add residual connection
        x = layers.Add()([lstm_out, x])
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(MODEL_CONFIG['model']['dropout_rate'])(x)
    
    # Attention layer
    attention = Attention()(x)
    
    # Dense layers
    x = attention
    for units in MODEL_CONFIG['model']['dense_units']:
        # Dense block
        dense = layers.Dense(
            units=units,
            kernel_regularizer=regularizers.l2(MODEL_CONFIG['model']['l2_reg'])
        )(x)
        bn = layers.BatchNormalization()(dense)
        x = layers.Activation('relu')(bn)
        x = layers.Dropout(MODEL_CONFIG['model']['dropout_rate'])(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def cosine_annealing_schedule(epoch, lr):
    """Cosine annealing learning rate schedule."""
    initial_lr = MODEL_CONFIG['training']['learning_rate']
    min_lr = MODEL_CONFIG['training']['reduce_lr_min_lr']
    max_epochs = MODEL_CONFIG['training']['epochs']
    
    # Warmup phase
    if epoch < MODEL_CONFIG['training']['warmup_epochs']:
        return initial_lr * (epoch + 1) / MODEL_CONFIG['training']['warmup_epochs']
    
    # Cosine annealing phase
    progress = (epoch - MODEL_CONFIG['training']['warmup_epochs']) / (max_epochs - MODEL_CONFIG['training']['warmup_epochs'])
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * progress))

def preprocess_features(features, labels=None, is_training=True):
    """Preprocess features with normalization and selection."""
    from sklearn.preprocessing import StandardScaler
    
    if MODEL_CONFIG['feature_extraction']['normalize_features']:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    
    return features

def get_latest_preprocessed_files():
    """Get the latest preprocessed features and labels files."""
    features_files = sorted(glob.glob('data/processed/features/features_*.npy'))
    labels_files = sorted(glob.glob('data/processed/labels/labels_*.npy'))
    if not features_files or not labels_files:
        raise FileNotFoundError('No preprocessed features or labels files found.')
    latest_features = features_files[-1]
    latest_labels = labels_files[-1]
    return latest_features, latest_labels

def load_model_with_custom_objects(model_path):
    """Load model with custom FocalLoss and Attention layers."""
    try:
        model = models.load_model(
            model_path,
            custom_objects={
                'FocalLoss': FocalLoss,
                'Attention': Attention
            }
        )
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        return None

def train_model():
    """Train the model using k-fold cross-validation."""
    try:
        # Load preprocessed data
        features_file, labels_file = get_latest_preprocessed_files()
        features = np.load(features_file)
        labels = np.load(labels_file)
        
        logger.info(f"Loaded features shape: {features.shape}")
        logger.info(f"Loaded labels shape: {labels.shape}")
        
        # Preprocess features
        features = preprocess_features(features, labels)
        
        # Encode labels
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        num_classes = len(label_encoder.classes_)
        
        logger.info(f"Number of classes: {num_classes}")
        logger.info(f"Class mapping: {dict(zip(label_encoder.classes_, range(num_classes)))}")
        
        # Compute class weights
        class_weights = compute_class_weights(labels_encoded)
        logger.info(f"Class weights: {class_weights}")
        
        # Initialize k-fold cross-validation
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Initialize lists to store results
        histories = []
        fold_metrics = []
        
        # Create output directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join('models', timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        # Train model for each fold
        for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels_encoded), 1):
            logger.info(f"\nTraining fold {fold}/{n_splits}")
            
            # Split data
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = labels_encoded[train_idx], labels_encoded[val_idx]
            
            # Create and compile model
            model = create_model(MODEL_CONFIG['model']['input_shape'], num_classes)
            
            # Use Focal Loss for training
            loss_fn = FocalLoss(
                gamma=MODEL_CONFIG['training']['focal_loss']['gamma'],
                alpha=MODEL_CONFIG['training']['focal_loss']['alpha']
            )
            
            model.compile(
                optimizer=Adam(learning_rate=MODEL_CONFIG['training']['learning_rate']),
                loss=loss_fn,
                metrics=['accuracy']
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=MODEL_CONFIG['training']['early_stopping_patience'],
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=MODEL_CONFIG['training']['reduce_lr_factor'],
                    patience=MODEL_CONFIG['training']['reduce_lr_patience'],
                    min_lr=MODEL_CONFIG['training']['reduce_lr_min_lr']
                ),
                ModelCheckpoint(
                    filepath=os.path.join(output_dir, f'model_fold_{fold}.h5'),
                    monitor='val_loss',
                    save_best_only=True
                )
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=MODEL_CONFIG['training']['epochs'],
                batch_size=MODEL_CONFIG['training']['batch_size'],
                callbacks=callbacks,
                class_weight=class_weights if MODEL_CONFIG['training']['class_weights'] else None,
                verbose=1
            )
            
            # Evaluate model
            metrics = evaluate_model(model, X_val, y_val, fold)
            fold_metrics.append(metrics)
            
            # Store history
            histories.append(history.history)
            
            # Save fold results
            fold_results = {
                'fold': fold,
                'metrics': metrics,
                'history': history.history
            }
            with open(os.path.join(output_dir, f'fold_{fold}_results.json'), 'w') as f:
                json.dump(fold_results, f, indent=4)
        
        # Calculate and save average metrics
        avg_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
            'precision': np.mean([m['precision'] for m in fold_metrics]),
            'recall': np.mean([m['recall'] for m in fold_metrics]),
            'f1_score': np.mean([m['f1_score'] for m in fold_metrics])
        }
        
        # Save training results
        results = {
            'timestamp': timestamp,
            'model_config': MODEL_CONFIG,
            'average_metrics': avg_metrics,
            'fold_metrics': fold_metrics
        }
        
        with open(os.path.join(output_dir, 'training_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        # Plot training history
        plot_training_history(histories, os.path.join(output_dir, 'training_history.png'))
        
        logger.info("\nTraining completed successfully!")
        logger.info(f"Average metrics across {n_splits} folds:")
        for metric, value in avg_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return output_dir
        
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test, fold=None):
    """Evaluate model performance."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred_classes),
        'precision': precision_score(y_test, y_pred_classes, average='weighted'),
        'recall': recall_score(y_test, y_pred_classes, average='weighted'),
        'f1_score': f1_score(y_test, y_pred_classes, average='weighted')
    }
    
    # Log metrics
    if fold:
        logger.info(f"\nFold {fold} metrics:")
    else:
        logger.info("\nTest metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix."""
    from sklearn.metrics import confusion_matrix
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path)
    plt.close()

def plot_training_history(histories, save_path):
    """Plot and save training history."""
    # Calculate average metrics across folds
    avg_loss = np.mean([h['loss'] for h in histories], axis=0)
    avg_val_loss = np.mean([h['val_loss'] for h in histories], axis=0)
    avg_acc = np.mean([h['accuracy'] for h in histories], axis=0)
    avg_val_acc = np.mean([h['val_accuracy'] for h in histories], axis=0)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Loss plot
    ax1.plot(avg_loss, label='Training Loss')
    ax1.plot(avg_val_loss, label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(avg_acc, label='Training Accuracy')
    ax2.plot(avg_val_acc, label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compute_class_weights(labels):
    """Compute class weights for imbalanced dataset."""
    from sklearn.utils.class_weight import compute_class_weight
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    return dict(zip(np.unique(labels), class_weights))

if __name__ == '__main__':
    train_model()