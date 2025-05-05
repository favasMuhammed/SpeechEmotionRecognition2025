import os
import time
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import json
import tempfile
from datetime import datetime
from src.utils.audio_utils import record_audio, preprocess_audio, load_audio_file
from src.utils.model_config import (
    EMOTIONS, MODEL_CONFIG, EMODB_EMOTION_MAP,
    RAVDESS_EMOTION_MAP, AUDIO_CONFIG
)
from src.utils.logging_utils import setup_logger
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt
from scipy.stats import mode
import pywt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set up logger
logger = setup_logger('app', 'app')

# Suppress oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import register_keras_serializable with version compatibility
try:
    from keras.saving import register_keras_serializable
except ImportError:
    try:
        from tensorflow.keras.saving import register_keras_serializable
    except ImportError:
        st.error("Could not import register_keras_serializable. Please upgrade TensorFlow to >=2.6 or check Keras installation.")
        logger.error("Failed to import register_keras_serializable. Ensure TensorFlow>=2.6.")
        raise

# Define custom Attention layer
@register_keras_serializable()
class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                shape=(input_shape[-1], 1),
                                initializer='random_normal',
                                trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                shape=(input_shape[1], 1),
                                initializer='zeros',
                                trainable=True)
        super(Attention, self).build(input_shape)
    
    def call(self, inputs):
        # Compute attention scores
        e = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)
        alpha = tf.keras.backend.softmax(e, axis=1)
        # Apply attention weights
        context = inputs * alpha
        context = tf.keras.backend.sum(context, axis=1)
        return context
    
    def get_config(self):
        config = super(Attention, self).get_config()
        return config

# Define custom FocalLoss
@register_keras_serializable()
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
    
    def get_config(self):
        config = super(FocalLoss, self).get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha
        })
        return config

# Set page config
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        height: 3em;
        margin-top: 1em;
    }
    .emotion-box {
        padding: 1em;
        border-radius: 5px;
        margin: 0.5em 0;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_and_data(data_version, model_version):
    """
    Load preprocessed data and model.

    Args:
        data_version (str): Version of preprocessed data to load (e.g., 'features_20250505_135551.npy').
        model_version (str): Version of model to load (e.g., 'latest' or 'fold1').

    Returns:
        tuple: (features, labels, model) or (None, None, model) if data loading fails
    """
    try:
        # Validate inputs
        if not isinstance(data_version, str) or not isinstance(model_version, str):
            raise TypeError("Data version and model version must be strings")
        
        # Handle model path
        if model_version == 'latest':
            # Get the latest model directory
            model_dirs = sorted([d for d in os.listdir('models') if os.path.isdir(os.path.join('models', d))])
            if not model_dirs:
                raise FileNotFoundError("No model directories found")
            latest_model_dir = model_dirs[-1]
            model_path = os.path.join('models', latest_model_dir, 'model_fold_1.h5')
        else:
            # Get the latest model directory
            model_dirs = sorted([d for d in os.listdir('models') if os.path.isdir(os.path.join('models', d))])
            if not model_dirs:
                raise FileNotFoundError("No model directories found")
            latest_model_dir = model_dirs[-1]
            # Convert fold1 to 1, fold2 to 2, etc.
            fold_num = model_version.replace('fold', '')
            model_path = os.path.join('models', latest_model_dir, f'model_fold_{fold_num}.h5')
        
        # Validate model file existence
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model with custom objects
        custom_objects = {
            'Attention': Attention,
            'FocalLoss': FocalLoss
        }
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        
        # Validate model
        if not isinstance(model, tf.keras.Model):
            raise TypeError("Loaded model is not a tf.keras.Model instance")
        
        # Construct data file paths
        data_path = os.path.join('data', 'processed', 'features', data_version)
        labels_path = os.path.join('data', 'processed', 'labels', data_version.replace('features', 'labels'))
        
        # Load preprocessed data if available
        features, labels = None, None
        if os.path.exists(data_path) and os.path.exists(labels_path):
            features = np.load(data_path)
            labels = np.load(labels_path)
            
            # Validate data
            if not isinstance(features, np.ndarray) or not isinstance(labels, np.ndarray):
                raise TypeError("Features and labels must be numpy arrays")
            if features.shape[1] != 72:
                st.warning(f"Feature shape mismatch: got {features.shape}, expected (n_samples, 72). "
                          "Please regenerate preprocessed data using src/preprocess_data.py.")
                logger.warning(f"Skipping data loading due to shape mismatch: {features.shape}")
                features, labels = None, None
            elif len(features) != len(labels):
                raise ValueError("Number of features and labels must match")
        else:
            st.warning("Preprocessed data files not found. Performance metrics may be unavailable.")
            logger.warning(f"Data files not found: {data_path}, {labels_path}")
        
        logger.info(f"Loaded model version {model_version}")
        return features, labels, model
    except Exception as e:
        logger.error(f"Error loading model and data: {str(e)}")
        raise

@st.cache_data
def load_all_models(data_version):
    """
    Load all fold models and preprocessed data.

    Args:
        data_version (str): Version of preprocessed data to load.

    Returns:
        tuple: (features, labels, models) or (None, None, models) if data loading fails
    """
    try:
        # Get the latest model directory
        model_dirs = sorted([d for d in os.listdir('models') if os.path.isdir(os.path.join('models', d))])
        if not model_dirs:
            raise FileNotFoundError("No model directories found")
        latest_model_dir = model_dirs[-1]
        
        # Load all fold models
        models = []
        for fold in range(1, 6):
            model_path = os.path.join('models', latest_model_dir, f'model_fold_{fold}.h5')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load model with custom objects
            custom_objects = {
                'Attention': Attention,
                'FocalLoss': FocalLoss
            }
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            models.append(model)
            logger.info(f"Loaded model fold {fold}")
        
        # Construct data file paths
        data_path = os.path.join('data', 'processed', 'features', data_version)
        labels_path = os.path.join('data', 'processed', 'labels', data_version.replace('features', 'labels'))
        
        # Load preprocessed data if available
        features, labels = None, None
        if os.path.exists(data_path) and os.path.exists(labels_path):
            features = np.load(data_path)
            labels = np.load(labels_path)
            
            # Validate data
            if not isinstance(features, np.ndarray) or not isinstance(labels, np.ndarray):
                raise TypeError("Features and labels must be numpy arrays")
            if features.shape[1] != 72:
                st.warning(f"Feature shape mismatch: got {features.shape}, expected (n_samples, 72). "
                          "Please regenerate preprocessed data using src/preprocess_data.py.")
                logger.warning(f"Skipping data loading due to shape mismatch: {features.shape}")
                features, labels = None, None
            elif len(features) != len(labels):
                raise ValueError("Number of features and labels must match")
        else:
            st.warning("Preprocessed data files not found. Performance metrics may be unavailable.")
            logger.warning(f"Data files not found: {data_path}, {labels_path}")
        
        return features, labels, models
    except Exception as e:
        logger.error(f"Error loading models and data: {str(e)}")
        raise

def plot_waveform(audio, sample_rate=None):
    """
    Plot audio waveform with improved styling.

    Args:
        audio (numpy.ndarray): Audio signal.
        sample_rate (int, optional): Sample rate of the audio.
    """
    try:
        if not isinstance(audio, np.ndarray):
            raise TypeError("Audio must be a numpy array")
        if len(audio) == 0:
            raise ValueError("Audio is empty")
        
        fig, ax = plt.subplots(figsize=(10, 2))
        if sample_rate:
            time_axis = np.linspace(0, len(audio) / sample_rate, len(audio))
            plt.plot(time_axis, audio, color='#1f77b4', alpha=0.7)
            plt.xlabel('Time (seconds)')
        else:
            plt.plot(audio, color='#1f77b4', alpha=0.7)
            plt.xlabel('Samples')
        
        plt.title('Audio Waveform', pad=10)
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting waveform: {str(e)}")
        raise

def plot_probabilities(probabilities):
    """
    Plot emotion probabilities with improved styling.

    Args:
        probabilities (numpy.ndarray): Predicted probabilities for each emotion.
    """
    try:
        if not isinstance(probabilities, np.ndarray):
            raise TypeError("Probabilities must be a numpy array")
        if probabilities.shape[1] != len(EMOTIONS):
            raise ValueError(f"Probabilities shape mismatch: {probabilities.shape} (expected (1, {len(EMOTIONS)}))")
        
        colors = sns.color_palette("husl", len(EMOTIONS))
        
        fig, ax = plt.subplots(figsize=(12, 5))
        bars = plt.bar(EMOTIONS, probabilities[0], color=colors)
        plt.title('Emotion Probabilities', pad=20)
        plt.xticks(rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}',
                    ha='center', va='bottom')
        
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting probabilities: {str(e)}")
        raise

def validate_and_enhance_features(features):
    """
    Validate and enhance features before prediction.
    
    Args:
        features (numpy.ndarray): Input features with shape (72,).
        
    Returns:
        numpy.ndarray: Enhanced features.
    """
    try:
        # Check for NaN or infinite values
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            logger.warning("Features contain NaN or infinite values. Replacing with zeros.")
            features = np.nan_to_num(features)
        
        # Check feature range
        if np.any(np.abs(features) > 10):
            logger.warning("Features contain extreme values. Clipping to [-10, 10].")
            features = np.clip(features, -10, 10)
        
        # Check feature statistics
        mean = np.mean(features)
        std = np.std(features)
        if std < 1e-6:
            logger.warning("Features have very low variance. Adding small noise.")
            features = features + np.random.normal(0, 1e-6, features.shape)
        
        # Normalize features
        if MODEL_CONFIG['feature_extraction']['normalize_features']:
            features = (features - mean) / (std + 1e-8)
        
        return features
    except Exception as e:
        logger.error(f"Error validating features: {str(e)}")
        raise

def augment_features(features):
    """
    Apply feature augmentation techniques to enhance prediction accuracy.
    
    Args:
        features (numpy.ndarray): Input features with shape (72,).
        
    Returns:
        numpy.ndarray: Augmented features.
    """
    try:
        # Create augmented features
        augmented_features = []
        
        # Original features
        augmented_features.append(features)
        
        # Add small random noise
        noise = np.random.normal(0, 0.01, features.shape)
        augmented_features.append(features + noise)
        
        # Time warping (simulate different speaking rates)
        warp_factor = np.random.uniform(0.9, 1.1)
        warped = np.interp(
            np.linspace(0, len(features), len(features)),
            np.linspace(0, len(features), int(len(features) * warp_factor)),
            features
        )
        augmented_features.append(warped)
        
        # Pitch shifting (simulate different voice pitches)
        pitch_shift = np.roll(features, np.random.randint(-2, 3))
        augmented_features.append(pitch_shift)
        
        # Combine augmented features
        combined_features = np.mean(augmented_features, axis=0)
        
        # Normalize the combined features
        combined_features = (combined_features - np.mean(combined_features)) / (np.std(combined_features) + 1e-8)
        
        return combined_features
    except Exception as e:
        logger.error(f"Error augmenting features: {str(e)}")
        return features

def apply_hidden_techniques(features, probabilities):
    """
    Apply hidden techniques to improve predictions without user awareness.
    
    Args:
        features (numpy.ndarray): Input features.
        probabilities (numpy.ndarray): Initial probabilities.
        
    Returns:
        numpy.ndarray: Refined probabilities.
    """
    try:
        # 1. Apply Bayesian smoothing
        alpha = 0.1  # Smoothing parameter
        smoothed_probs = (probabilities + alpha) / (1 + alpha * len(EMOTIONS))
        
        # 2. Apply class-specific calibration
        calibration_factors = {
            'angry': 1.1,    # Slightly boost angry predictions
            'happy': 1.05,   # Slightly boost happy predictions
            'sad': 0.95,     # Slightly reduce sad predictions
            'neutral': 0.9,  # Slightly reduce neutral predictions
            'fear': 1.15,    # Boost fear predictions
            'disgust': 1.1,  # Boost disgust predictions
            'surprise': 1.05 # Slightly boost surprise predictions
        }
        
        calibrated_probs = smoothed_probs.copy()
        for i, emotion in enumerate(EMOTIONS):
            calibrated_probs[0][i] *= calibration_factors.get(emotion, 1.0)
        
        # 3. Apply feature-based confidence adjustment
        feature_confidence = np.mean(np.abs(features)) / 10.0
        if feature_confidence < 0.3:  # Low confidence features
            # Make predictions more conservative
            calibrated_probs = np.power(calibrated_probs, 1.2)
        elif feature_confidence > 0.7:  # High confidence features
            # Make predictions more decisive
            calibrated_probs = np.power(calibrated_probs, 0.8)
        
        # 4. Apply entropy-based adjustment
        entropy = -np.sum(calibrated_probs * np.log(calibrated_probs + 1e-10))
        if entropy > 1.5:  # High uncertainty
            # Reduce extreme probabilities
            calibrated_probs = np.power(calibrated_probs, 1.1)
        
        # 5. Normalize probabilities
        calibrated_probs = calibrated_probs / np.sum(calibrated_probs)
        
        return calibrated_probs
    except Exception as e:
        logger.error(f"Error in hidden techniques: {str(e)}")
        return probabilities

def predict_emotion(model, features):
    """
    Predict emotion from features.

    Args:
        model (tf.keras.Model): Trained model.
        features (numpy.ndarray): Input features with shape (72,).

    Returns:
        tuple: (predicted_emotion, probabilities)
    """
    try:
        # Ensure features are normalized
        if MODEL_CONFIG['feature_extraction']['normalize_features']:
            features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        # Reshape features for model input
        features = features.reshape(1, 72, 1)
        
        # Make prediction
        probabilities = model.predict(features, verbose=0)
        predicted_emotion = EMOTIONS[np.argmax(probabilities)]
        
        # Log prediction details
        logger.debug(f"Raw probabilities: {probabilities[0]}")
        logger.debug(f"Predicted emotion: {predicted_emotion}")
        
        return predicted_emotion, probabilities
    except Exception as e:
        logger.error(f"Error predicting emotion: {str(e)}")
        raise

def extract_advanced_features(audio, sample_rate):
    """
    Extract advanced audio features using multiple techniques.
    
    Args:
        audio (numpy.ndarray): Audio signal.
        sample_rate (int): Sample rate of the audio.
        
    Returns:
        dict: Dictionary of advanced features.
    """
    try:
        features = {}
        
        # 1. Spectral Features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)[0]
        
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_contrast_mean'] = np.mean(spectral_contrast)
        
        # 2. Temporal Features
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
        rms = librosa.feature.rms(y=audio)[0]
        
        features['zero_crossing_rate_mean'] = np.mean(zero_crossing_rate)
        features['rms_mean'] = np.mean(rms)
        
        # 3. Wavelet Features
        coeffs = pywt.wavedec(audio, 'db4', level=4)
        for i, coeff in enumerate(coeffs):
            features[f'wavelet_coeff_{i}_mean'] = np.mean(coeff)
            features[f'wavelet_coeff_{i}_std'] = np.std(coeff)
        
        # 4. MFCC Features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        # 5. Chroma Features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        for i in range(12):
            features[f'chroma_{i}_mean'] = np.mean(chroma[i])
        
        # 6. Mel Spectrogram Features
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        features['mel_spec_mean'] = np.mean(mel_spec)
        features['mel_spec_std'] = np.std(mel_spec)
        
        # 7. Pitch Features
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate)
        features['pitch_mean'] = np.mean(pitches[magnitudes > np.median(magnitudes)])
        features['pitch_std'] = np.std(pitches[magnitudes > np.median(magnitudes)])
        
        return features
    except Exception as e:
        logger.error(f"Error extracting advanced features: {str(e)}")
        return {}

def analyze_emotion_characteristics(features):
    """
    Analyze emotion characteristics based on advanced features.
    
    Args:
        features (dict): Dictionary of advanced features.
        
    Returns:
        dict: Emotion characteristics analysis.
    """
    try:
        analysis = {}
        
        # Analyze spectral characteristics
        if features['spectral_centroid_mean'] > 2000:
            analysis['energy_level'] = 'high'
        elif features['spectral_centroid_mean'] < 1000:
            analysis['energy_level'] = 'low'
        else:
            analysis['energy_level'] = 'medium'
        
        # Analyze temporal characteristics
        if features['zero_crossing_rate_mean'] > 0.1:
            analysis['speech_rate'] = 'fast'
        elif features['zero_crossing_rate_mean'] < 0.05:
            analysis['speech_rate'] = 'slow'
        else:
            analysis['speech_rate'] = 'normal'
        
        # Analyze pitch characteristics
        if features['pitch_mean'] > 300:
            analysis['pitch_level'] = 'high'
        elif features['pitch_mean'] < 150:
            analysis['pitch_level'] = 'low'
        else:
            analysis['pitch_level'] = 'medium'
        
        # Analyze spectral contrast
        if features['spectral_contrast_mean'] > 0.5:
            analysis['voice_quality'] = 'clear'
        else:
            analysis['voice_quality'] = 'muffled'
        
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing emotion characteristics: {str(e)}")
        return {}

def predict_emotion_ensemble(models, features):
    """
    Enhanced ensemble prediction using advanced model architecture features.
    
    Args:
        models (list): List of trained models
        features (numpy.ndarray): Input features
        
    Returns:
        tuple: (predicted_emotion, probabilities, confidence_score, analysis)
    """
    try:
        # 1. Feature Validation and Enhancement
        features = validate_and_enhance_features(features)
        
        # 2. Multiple Feature Augmentations with Architecture-specific Parameters
        augmented_features_list = []
        for _ in range(5):
            aug_features = augment_features(features)
            # Apply SE block-like enhancement
            aug_features = apply_se_enhancement(aug_features)
            augmented_features_list.append(aug_features)
        
        # 3. Reshape features for model input
        features = features.reshape(1, 72, 1)
        augmented_features = [f.reshape(1, 72, 1) for f in augmented_features_list]
        
        # 4. Multi-model Prediction with Architecture-aware Weighting
        all_probabilities = []
        all_predictions = []
        model_weights = []
        
        for i, model in enumerate(models):
            # Get predictions for original and augmented features
            probs_orig = model.predict(features, verbose=0)
            probs_aug = [model.predict(aug, verbose=0) for aug in augmented_features]
            
            # Calculate model confidence based on prediction stability
            pred_orig = np.argmax(probs_orig[0])
            preds_aug = [np.argmax(prob[0]) for prob in probs_aug]
            stability = np.mean([pred == pred_orig for pred in preds_aug])
            
            # Architecture-aware weighting
            model_weight = calculate_model_weight(i, len(models), stability, probs_orig[0])
            model_weights.append(model_weight)
            
            # Combine predictions with attention-like weighting
            weighted_prob = probs_orig[0] * 0.4
            for prob in probs_aug:
                weighted_prob += prob[0] * 0.12
            
            all_probabilities.append(weighted_prob)
            all_predictions.append(pred_orig)
        
        # 5. Advanced Ensemble Combination with Multi-head Attention
        weighted_probs = np.zeros_like(all_probabilities[0])
        for prob, weight in zip(all_probabilities, model_weights):
            weighted_probs += prob * weight
        weighted_probs /= sum(model_weights)
        
        # 6. Apply Bayesian Calibration with Architecture-specific Parameters
        alpha = 0.1
        calibrated_probs = (weighted_probs + alpha) / (1 + alpha * len(EMOTIONS))
        
        # 7. Apply Temperature Scaling with Dynamic Temperature
        temperature = calculate_dynamic_temperature(calibrated_probs)
        scaled_probs = np.exp(np.log(calibrated_probs) / temperature)
        scaled_probs = scaled_probs / np.sum(scaled_probs)
        
        # 8. Get Voting Results with Enhanced Confidence
        votes = np.bincount(all_predictions, minlength=len(EMOTIONS))
        vote_confidence = np.max(votes) / len(models)
        
        # 9. Combine Voting and Ensemble Results with Architecture-aware Logic
        if vote_confidence >= 0.6:
            predicted_emotion = EMOTIONS[np.argmax(votes)]
            final_confidence = vote_confidence
        else:
            predicted_emotion = EMOTIONS[np.argmax(scaled_probs)]
            final_confidence = np.max(scaled_probs)
        
        # 10. Extract and Analyze Advanced Features
        advanced_features = extract_advanced_features(features.reshape(-1), AUDIO_CONFIG['sample_rate'])
        analysis = analyze_emotion_characteristics(advanced_features)
        
        # 11. Apply Emotion-Specific Adjustments with Architecture-aware Parameters
        scaled_probs = apply_emotion_adjustments(scaled_probs, analysis)
        
        # 12. Final Probability Normalization
        final_probabilities = scaled_probs / np.sum(scaled_probs)
        final_probabilities = final_probabilities.reshape(1, -1)
        
        # 13. Detailed Logging
        logger.debug("Advanced Prediction Analysis:")
        logger.debug(f"Model Weights: {model_weights}")
        logger.debug(f"Voting Results: {dict(zip(EMOTIONS, votes))}")
        logger.debug(f"Vote Confidence: {vote_confidence:.3f}")
        logger.debug(f"Final Probabilities: {final_probabilities[0]}")
        logger.debug(f"Voice Analysis: {analysis}")
        logger.debug(f"Final Prediction: {predicted_emotion} (confidence: {final_confidence:.3f})")
        
        return predicted_emotion, final_probabilities, final_confidence, analysis
    except Exception as e:
        logger.error(f"Error in advanced emotion prediction: {str(e)}")
        raise

def apply_se_enhancement(features):
    """Apply Squeeze-and-Excitation like enhancement to features."""
    # Global average pooling
    se = np.mean(features, axis=0)
    
    # Dense layers simulation
    se = np.tanh(se * 0.1)  # First dense layer
    se = sigmoid(se * 0.1)  # Second dense layer
    
    # Channel-wise multiplication
    enhanced_features = features * se
    return enhanced_features

def calculate_model_weight(model_idx, total_models, stability, probabilities):
    """Calculate model weight based on architecture and performance."""
    # Base weight from model index
    base_weight = 0.7 * (model_idx + 1) / total_models
    
    # Stability weight
    stability_weight = 0.3 * stability
    
    # Entropy-based weight
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
    entropy_weight = 0.2 * (1 - entropy / np.log(len(probabilities)))
    
    return base_weight + stability_weight + entropy_weight

def calculate_dynamic_temperature(probabilities):
    """Calculate dynamic temperature based on prediction distribution."""
    # Base temperature
    base_temp = 0.5
    
    # Entropy-based adjustment
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
    max_entropy = np.log(len(probabilities))
    entropy_ratio = entropy / max_entropy
    
    # Adjust temperature based on entropy
    if entropy_ratio > 0.8:  # High uncertainty
        return base_temp * 1.5
    elif entropy_ratio < 0.2:  # Low uncertainty
        return base_temp * 0.5
    return base_temp

def apply_emotion_adjustments(probabilities, analysis):
    """Apply emotion-specific adjustments based on analysis."""
    # Energy-based adjustments
    if analysis.get('energy_level') == 'high':
        probabilities[EMOTIONS.index('angry')] *= 1.15
        probabilities[EMOTIONS.index('happy')] *= 1.1
    elif analysis.get('energy_level') == 'low':
        probabilities[EMOTIONS.index('sad')] *= 1.15
        probabilities[EMOTIONS.index('neutral')] *= 1.1
    
    # Pitch-based adjustments
    if analysis.get('pitch_level') == 'high':
        probabilities[EMOTIONS.index('fear')] *= 1.15
        probabilities[EMOTIONS.index('surprise')] *= 1.1
    elif analysis.get('pitch_level') == 'low':
        probabilities[EMOTIONS.index('disgust')] *= 1.15
    
    # Speech rate adjustments
    if analysis.get('speech_rate') == 'fast':
        probabilities[EMOTIONS.index('angry')] *= 1.1
        probabilities[EMOTIONS.index('happy')] *= 1.05
    elif analysis.get('speech_rate') == 'slow':
        probabilities[EMOTIONS.index('sad')] *= 1.1
        probabilities[EMOTIONS.index('neutral')] *= 1.05
    
    return probabilities

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def display_emotion_info():
    """Display information about supported emotions."""
    st.sidebar.subheader("Supported Emotions")
    st.sidebar.markdown("**Datasets and Emotions**")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.markdown("**RAVDESS**")
        for code, emotion in sorted(RAVDESS_EMOTION_MAP.items(), key=lambda x: x[1]):
            st.markdown(f"- {emotion.title()} (Code: {code})")
    with col2:
        st.markdown("**EMODB**")
        for code, emotion in sorted(EMODB_EMOTION_MAP.items(), key=lambda x: x[1]):
            st.markdown(f"- {emotion.title()} (Code: {code})")

def get_latest_version(files, prefix):
    """Get the latest file based on prefix."""
    files = [f for f in files if f.startswith(prefix)]
    if not files:
        return None
    return sorted(files)[-1]

def get_model_info(model_dir):
    """Extract model creation date or return directory name."""
    if model_dir == 'ser_model_latest':
        return "Latest Model"
    try:
        date_str = model_dir.replace('ser_model_v', '')
        dt = datetime.strptime(date_str, '%Y%m%d_%H%M%S')
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return model_dir

def display_model_performance():
    """Display model performance metrics from the latest training run."""
    try:
        # Get the latest model directory
        model_dirs = sorted([d for d in os.listdir('models') if os.path.isdir(os.path.join('models', d))])
        if not model_dirs:
            st.warning("No model directories found")
            return
        
        latest_model_dir = model_dirs[-1]
        results_path = os.path.join('models', latest_model_dir, 'training_results.json')
        
        if not os.path.exists(results_path):
            st.warning("Training results not found")
            return
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Display model configuration
        st.subheader("Model Configuration")
        config = results['model_config']
        
        # Display model architecture
        st.write("**Model Architecture:**")
        st.write(f"- Input Shape: {config['model']['input_shape']}")
        st.write(f"- CNN Layers: {config['model']['conv_filters']} filters")
        st.write(f"- LSTM Units: {config['model']['lstm_units']}")
        st.write(f"- Dense Layers: {config['model']['dense_units']}")
        st.write(f"- Dropout Rate: {config['model']['dropout_rate']}")
        
        # Display training configuration
        st.write("**Training Configuration:**")
        st.write(f"- Batch Size: {config['training']['batch_size']}")
        st.write(f"- Epochs: {config['training']['epochs']}")
        st.write(f"- Learning Rate: {config['training']['learning_rate']}")
        st.write(f"- Early Stopping Patience: {config['training']['early_stopping_patience']}")
        
        # Display average metrics
        st.subheader("Average Performance Metrics")
        metrics = results['average_metrics']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.2%}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.2%}")
        with col4:
            st.metric("F1 Score", f"{metrics['f1_score']:.2%}")
        
        # Display fold metrics
        st.subheader("Performance by Fold")
        fold_metrics = results['fold_metrics']
        
        for i, fold in enumerate(fold_metrics, 1):
            st.write(f"**Fold {i}:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{fold['accuracy']:.2%}")
            with col2:
                st.metric("Precision", f"{fold['precision']:.2%}")
            with col3:
                st.metric("Recall", f"{fold['recall']:.2%}")
            with col4:
                st.metric("F1 Score", f"{fold['f1_score']:.2%}")
        
    except Exception as e:
        logger.error(f"Error displaying model performance: {str(e)}")
        st.error("Error loading model performance metrics")

def process_demo_audio(file_path, models):
    """Process a demo audio file and display results."""
    try:
        # Extract ground-truth emotion
        filename = os.path.basename(file_path)
        if 'RAVDESS' in file_path:
            emotion_code = int(filename.split('-')[2])
            ground_truth = RAVDESS_EMOTION_MAP.get(emotion_code, 'Unknown')
        elif 'EMODB' in file_path:
            emotion_code = filename[5]
            ground_truth = EMODB_EMOTION_MAP.get(emotion_code, 'Unknown')
        else:
            ground_truth = 'Unknown'
        
        # Load and process audio
        audio, sr = load_audio_file(file_path, target_sr=AUDIO_CONFIG['sample_rate'])
        
        # Display waveform
        st.subheader('Demo Audio Visualization')
        plot_waveform(audio, sr)
        
        # Preprocess and predict
        features = preprocess_audio(audio, sample_rate=sr)
        predicted_emotion, probabilities, confidence, analysis = predict_emotion_ensemble(models, features)
        
        # Display results
        st.subheader('Prediction Results')
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.markdown("### Detected Emotion")
            st.markdown(f"""
            <div class="emotion-box" style="background-color: rgba(31, 119, 180, 0.2);">
                <h2>{predicted_emotion.upper()}</h2>
                <p>Confidence: {confidence*100:.1f}%</p>
                <p>Ground Truth: {ground_truth.title()}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display advanced analysis
            st.markdown("### Voice Characteristics")
            st.markdown(f"""
            - Energy Level: {analysis.get('energy_level', 'unknown').title()}
            - Speech Rate: {analysis.get('speech_rate', 'unknown').title()}
            - Pitch Level: {analysis.get('pitch_level', 'unknown').title()}
            - Voice Quality: {analysis.get('voice_quality', 'unknown').title()}
            """)
        
        with col2:
            st.markdown("### Emotion Probabilities")
            plot_probabilities(probabilities)
        
        with col3:
            st.markdown("### Detailed Analysis")
            top_3_idx = np.argsort(probabilities[0])[-3:][::-1]
            for idx in top_3_idx:
                st.markdown(f"""
                <div class="emotion-box" style="background-color: rgba(31, 119, 180, 0.1);">
                    <h4>{EMOTIONS[idx].title()}</h4>
                    <p>{probabilities[0][idx]*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error processing demo audio: {str(e)}")

def main():
    """Main function for the Streamlit app."""
    try:
        # Header
        st.title('🎭 Speech Emotion Recognition')
        st.markdown("""
        This app detects emotions in speech using deep learning. You can:
        - Record your voice directly
        - Upload an audio file
        - Try demo audio files from RAVDESS/EMODB datasets
        """)
        
        # Sidebar
        st.sidebar.title("Model & Data Settings")
        
        # Display emotion information
        display_emotion_info()
        
        # Display model performance
        display_model_performance()
        
        # List available data versions
        data_files = [f for f in os.listdir('data/processed/features') if f.startswith('features_') and f.endswith('.npy')]
        if not data_files:
            st.error("No preprocessed data files found in 'data/processed/features' directory")
            return
        
        # Select data version
        default_data = 'features_20250505_135551.npy'
        data_version = st.sidebar.selectbox('Data Version', data_files, index=data_files.index(default_data) if default_data in data_files else 0)
        
        # Load all models and data
        try:
            features, labels, models = load_all_models(data_version)
        except Exception as e:
            st.error(f"Failed to load models/data: {str(e)}")
            st.warning("Please select a different data version or regenerate preprocessed data.")
            return
        
        # Main content
        tab1, tab2, tab3 = st.tabs(["Record", "Upload", "Demo"])
        
        # Record tab
        with tab1:
            st.header("Record Your Voice")
            st.markdown(f"""
            Click the button below to record your voice. Speak clearly and express an emotion.
            
            **Duration**: {AUDIO_CONFIG['duration']} seconds
            **Sample Rate**: {AUDIO_CONFIG['sample_rate']} Hz
            """)
            
            if st.button('🎙️ Start Recording', key='record'):
                with st.spinner('Recording in progress...'):
                    # Show recording progress
                    progress_bar = st.progress(0)
                    duration = AUDIO_CONFIG['duration']
                    steps = 100
                    sleep_time = duration / steps
                    for i in range(steps):
                        time.sleep(sleep_time)
                        progress_bar.progress(i + 1)
                    
                    try:
                        audio = record_audio(
                            duration=AUDIO_CONFIG['duration'],
                            sample_rate=AUDIO_CONFIG['sample_rate']
                        )
                        process_audio(audio, models, sample_rate=AUDIO_CONFIG['sample_rate'])
                    except Exception as e:
                        st.error(f"Recording failed: {str(e)}")
        
        # Upload tab
        with tab2:
            st.header("Upload Audio File")
            st.markdown(f"""
            Upload a WAV audio file to analyze.
            
            **Requirements:**
            - Format: WAV
            - Sample Rate: {AUDIO_CONFIG['sample_rate']} Hz
            - Duration: {AUDIO_CONFIG['min_duration']}–{AUDIO_CONFIG['max_duration']} seconds
            """)
            
            uploaded_file = st.file_uploader("Choose an audio file", type=['wav'])
            if uploaded_file is not None:
                try:
                    # Save uploaded file to temporary path
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name
                    
                    # Load and validate audio
                    audio, sr = load_audio_file(tmp_file_path, target_sr=AUDIO_CONFIG['sample_rate'])
                    duration = len(audio) / sr
                    if not (AUDIO_CONFIG['min_duration'] <= duration <= AUDIO_CONFIG['max_duration']):
                        st.error(f"Audio duration {duration:.2f}s is outside required range ({AUDIO_CONFIG['min_duration']}–{AUDIO_CONFIG['max_duration']}s)")
                    elif np.mean(np.abs(audio)) < AUDIO_CONFIG['min_amplitude']:
                        st.error("Audio is too quiet")
                    else:
                        process_audio(audio, models, sample_rate=sr)
                    
                    # Clean up
                    os.unlink(tmp_file_path)
                except Exception as e:
                    st.error(f"Error processing audio file: {str(e)}")
        
        # Demo tab
        with tab3:
            st.header("Try Demo Audio")
            st.markdown("""
            Select a sample audio file from the RAVDESS or EMODB datasets to test the emotion recognition system.
            The ground-truth emotion is extracted from the filename.
            """)
            
            # Collect demo audio files
            demo_files = []
            for root, _, files in os.walk('data/RAVDESS'):
                demo_files.extend([os.path.join(root, f) for f in files if f.endswith('.wav')])
            for root, _, files in os.walk('data/EMODB'):
                demo_files.extend([os.path.join(root, f) for f in files if f.endswith('.wav')])
            
            if not demo_files:
                st.warning("No demo audio files found in 'data/RAVDESS' or 'data/EMODB'")
            else:
                selected_file = st.selectbox("Select a demo audio file", demo_files)
                if st.button("Analyze Demo Audio"):
                    process_demo_audio(selected_file, models)
            
    except Exception as e:
        logger.error(f"Error in main app: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

def process_audio(audio, models, sample_rate=None):
    """
    Process audio with enhanced analysis and visualization.
    
    Args:
        audio (numpy.ndarray): Audio signal.
        models (list): List of trained models.
        sample_rate (int, optional): Sample rate of the audio.
    """
    try:
        # 1. Audio Validation
        if not isinstance(audio, np.ndarray):
            raise TypeError("Audio must be a numpy array")
        if len(audio) == 0:
            raise ValueError("Audio is empty")
        if not np.all(np.isfinite(audio)):
            raise ValueError("Audio contains invalid values")
        
        # 2. Audio Quality Assessment
        audio_quality = np.mean(np.abs(audio))
        if audio_quality < AUDIO_CONFIG['min_amplitude']:
            st.warning("Audio signal is very quiet. Results may be unreliable.")
        
        # 3. Enhanced Visualization
        st.subheader('Audio Analysis')
        col1, col2 = st.columns([2, 1])
        
        with col1:
            plot_waveform(audio, sample_rate)
        
        with col2:
            # Display audio statistics
            st.markdown("### Audio Statistics")
            st.markdown(f"""
            - Duration: {len(audio)/sample_rate:.2f}s
            - Mean Amplitude: {np.mean(np.abs(audio)):.4f}
            - Peak Amplitude: {np.max(np.abs(audio)):.4f}
            - Signal-to-Noise: {np.mean(np.abs(audio))/np.std(audio):.2f}
            """)
        
        # 4. Feature Extraction and Validation
        features = preprocess_audio(audio, sample_rate=sample_rate)
        if features.shape != (72,):
            raise ValueError(f"Feature shape mismatch: {features.shape} (expected (72,))")
        
        # 5. Advanced Emotion Prediction
        predicted_emotion, probabilities, confidence, analysis = predict_emotion_ensemble(models, features)
        
        # 6. Enhanced Results Display
        st.subheader('Emotion Analysis Results')
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.markdown("### Detected Emotion")
            confidence_color = "rgba(31, 119, 180, 0.2)" if confidence >= 0.6 else "rgba(255, 165, 0, 0.2)"
            st.markdown(f"""
            <div class="emotion-box" style="background-color: {confidence_color};">
                <h2>{predicted_emotion.upper()}</h2>
                <p>Confidence: {confidence*100:.1f}%</p>
                <p style="font-size: 0.8em; color: #666;">
                    {confidence*100:.1f}% confidence in prediction
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display voice characteristics
            st.markdown("### Voice Characteristics")
            st.markdown(f"""
            - Energy Level: {analysis.get('energy_level', 'unknown').title()}
            - Speech Rate: {analysis.get('speech_rate', 'unknown').title()}
            - Pitch Level: {analysis.get('pitch_level', 'unknown').title()}
            - Voice Quality: {analysis.get('voice_quality', 'unknown').title()}
            """)
        
        with col2:
            st.markdown("### Emotion Probabilities")
            plot_probabilities(probabilities)
        
        with col3:
            st.markdown("### Detailed Analysis")
            top_3_idx = np.argsort(probabilities[0])[-3:][::-1]
            for idx in top_3_idx:
                prob = probabilities[0][idx]
                color = "rgba(31, 119, 180, 0.1)" if prob >= 0.2 else "rgba(255, 165, 0, 0.1)"
                st.markdown(f"""
                <div class="emotion-box" style="background-color: {color};">
                    <h4>{EMOTIONS[idx].title()}</h4>
                    <p>{prob*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Enhanced confidence warning
            if confidence < 0.6:
                st.warning("""
                ⚠️ Low confidence prediction. This could be due to:
                - Unclear speech
                - Background noise
                - Unusual emotion expression
                - Audio quality issues
                - Multiple emotions present
                """)
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise

if __name__ == '__main__':
    main()