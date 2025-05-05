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
    """Create an enhanced CNN-Bidirectional LSTM model with multi-head attention and SE blocks."""
    # Input layer
    inputs = layers.Input(shape=input_shape)
    x = layers.Reshape((input_shape[0], 1))(inputs)
    
    # Squeeze-and-Excitation block
    def se_block(x, ratio=MODEL_CONFIG['model']['se_ratio']):
        filters = x.shape[-1]
        se = layers.GlobalAveragePooling1D()(x)
        se = layers.Dense(filters // ratio, activation='relu')(se)
        se = layers.Dense(filters, activation='sigmoid')(se)
        return layers.Multiply()([x, se])
    
    # Multi-head attention block
    def multi_head_attention(x, num_heads=MODEL_CONFIG['model']['attention_heads']):
        # Split into heads
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        depth = x.shape[-1]
        head_depth = depth // num_heads
        
        # Reshape for multi-head attention
        x = layers.Reshape((seq_len, num_heads, head_depth))(x)
        x = layers.Permute((2, 1, 3))(x)
        
        # Self-attention
        attention = layers.Dot(axes=(2, 2))([x, x])
        attention = layers.Softmax(axis=-1)(attention)
        
        # Apply attention
        context = layers.Dot(axes=(2, 1))([attention, x])
        context = layers.Permute((2, 1, 3))(context)
        context = layers.Reshape((seq_len, depth))(context)
        
        return context
    
    # CNN layers with residual connections and SE blocks
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
        
        # Residual connection
        if MODEL_CONFIG['model']['residual_connections']:
            if x.shape[-1] != filters:
                x = layers.Conv1D(filters=filters, kernel_size=1, padding='same')(x)
            x = layers.Add()([bn2, x])
        else:
            x = bn2
        
        # SE block
        x = se_block(x)
        x = layers.Activation('relu')(x)
        
        # Pooling and dropout
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(MODEL_CONFIG['model']['dropout_rate'])(x)
    
    # Bidirectional LSTM layers with residual connections
    for units in MODEL_CONFIG['model']['lstm_units']:
        # Main LSTM
        lstm_out = layers.Bidirectional(
            layers.LSTM(
                units=units,
                return_sequences=True,
                kernel_regularizer=regularizers.l2(MODEL_CONFIG['model']['l2_reg'])
            )
        )(x)
        
        # Residual connection
        if MODEL_CONFIG['model']['residual_connections']:
            if x.shape[-1] != lstm_out.shape[-1]:
                x = layers.Dense(lstm_out.shape[-1])(x)
            x = layers.Add()([lstm_out, x])
        else:
            x = lstm_out
        
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(MODEL_CONFIG['model']['dropout_rate'])(x)
    
    # Multi-head attention
    x = multi_head_attention(x)
    
    # Dense layers with residual connections
    for units in MODEL_CONFIG['model']['dense_units']:
        # Dense block
        dense = layers.Dense(
            units=units,
            kernel_regularizer=regularizers.l2(MODEL_CONFIG['model']['l2_reg'])
        )(x)
        bn = layers.BatchNormalization()(dense)
        x = layers.Activation('relu')(bn)
        x = layers.Dropout(MODEL_CONFIG['model']['dropout_rate'])(x)
    
    # Output layer with label smoothing
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        kernel_regularizer=regularizers.l2(MODEL_CONFIG['model']['l2_reg'])
    )(x)
    
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
    """Train the model using k-fold cross-validation with enhanced techniques."""
    try:
        # Load and preprocess data
        features_file, labels_file = get_latest_preprocessed_files()
        features = np.load(features_file)
        labels = np.load(labels_file)
        
        # Apply mixup augmentation
        if MODEL_CONFIG['training']['mixup_alpha'] > 0:
            features, labels = apply_mixup(features, labels, MODEL_CONFIG['training']['mixup_alpha'])
        
        # Preprocess features
        features = preprocess_features(features, labels)
        
        # Encode labels
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        num_classes = len(label_encoder.classes_)
        
        # Compute class weights
        class_weights = compute_class_weights(labels_encoded)
        
        # Initialize k-fold cross-validation
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Training loop
        histories = []
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels_encoded), 1):
            # Split data
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = labels_encoded[train_idx], labels_encoded[val_idx]
            
            # Create and compile model
            model = create_model(MODEL_CONFIG['model']['input_shape'], num_classes)
            
            # Use Focal Loss with label smoothing
            loss_fn = FocalLoss(
                gamma=MODEL_CONFIG['training']['focal_loss']['gamma'],
                alpha=MODEL_CONFIG['training']['focal_loss']['alpha']
            )
            
            # Optimizer with gradient clipping
            optimizer = Adam(
                learning_rate=MODEL_CONFIG['training']['learning_rate'],
                clipnorm=MODEL_CONFIG['training']['gradient_clip_norm']
            )
            
            model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=['accuracy']
            )
            
            # Enhanced callbacks
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
            
            # Evaluate and store results
            metrics = evaluate_model(model, X_val, y_val, fold)
            fold_metrics.append(metrics)
            histories.append(history.history)
        
        return fold_metrics, histories
        
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

def apply_mixup(features, labels, alpha=0.2):
    """Apply mixup augmentation to the dataset."""
    if alpha <= 0:
        return features, labels
    
    # Generate mixing weights
    lam = np.random.beta(alpha, alpha, size=len(features))
    
    # Create mixed features and labels
    mixed_features = np.zeros_like(features)
    mixed_labels = np.zeros_like(labels)
    
    for i in range(len(features)):
        # Randomly select another sample
        j = np.random.randint(0, len(features))
        
        # Mix features
        mixed_features[i] = lam[i] * features[i] + (1 - lam[i]) * features[j]
        
        # Mix labels (one-hot encoded)
        mixed_labels[i] = lam[i] * labels[i] + (1 - lam[i]) * labels[j]
    
    return mixed_features, mixed_labels

if __name__ == '__main__':
    train_model()