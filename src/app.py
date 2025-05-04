"""
Streamlit app for real-time emotion detection.
"""

import os
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from src.utils.audio_utils import record_audio, preprocess_audio
from src.utils.model_config import EMOTIONS, FEATURE_CONFIG
from src.utils.logging_utils import setup_logger

# Set up logger
logger = setup_logger('app', 'app')

def load_model_and_data(data_version, model_version):
    """
    Load preprocessed data and model.

    Args:
        data_version (str): Version of preprocessed data to load.
        model_version (str): Version of model to load.

    Returns:
        tuple: (features, labels, model)
    """
    try:
        # Load preprocessed data
        features = np.load(f'data/{data_version}')
        labels = np.load(f'data/labels_{data_version.split("features_")[1]}')
        
        # Load model
        model_path = f'models/ser_model_v{model_version}'
        model = tf.keras.models.load_model(model_path)
        
        logger.info(f"Loaded data version {data_version} and model version {model_version}")
        return features, labels, model
    except Exception as e:
        logger.error(f"Error loading model and data: {str(e)}")
        raise

def plot_waveform(audio):
    """
    Plot audio waveform.

    Args:
        audio (numpy.ndarray): Audio signal.
    """
    try:
        plt.figure(figsize=(10, 2))
        plt.plot(audio)
        plt.title('Audio Waveform')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        st.pyplot(plt.gcf())
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting waveform: {str(e)}")
        raise

def plot_probabilities(probabilities):
    """
    Plot emotion probabilities.

    Args:
        probabilities (numpy.ndarray): Predicted probabilities for each emotion.
    """
    try:
        plt.figure(figsize=(10, 4))
        plt.bar(EMOTIONS, probabilities[0])
        plt.title('Emotion Probabilities')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting probabilities: {str(e)}")
        raise

def main():
    """Main function for the Streamlit app."""
    try:
        st.title('Speech Emotion Recognition')
        
        # Sidebar for model and data selection
        st.sidebar.header('Settings')
        data_files = [f for f in os.listdir('data') if f.startswith('features_')]
        model_files = [f for f in os.listdir('models') if f.endswith('.h5')]
        
        data_version = st.sidebar.selectbox('Select Data Version', data_files)
        model_version = st.sidebar.selectbox('Select Model Version', model_files)
        
        # Load model and data
        features, labels, model = load_model_and_data(data_version, model_version)
        
        # Record button
        if st.button('Record'):
            st.write('Recording...')
            
            # Record and preprocess audio
            audio = record_audio(
                duration=FEATURE_CONFIG['duration'],
                sample_rate=FEATURE_CONFIG['sample_rate']
            )
            features = preprocess_audio(audio)
            
            # Plot waveform
            st.subheader('Audio Waveform')
            plot_waveform(audio)
            
            # Make prediction
            probabilities = model.predict(features)
            predicted_emotion = EMOTIONS[np.argmax(probabilities)]
            
            # Display results
            st.subheader('Prediction')
            st.write(f'Predicted Emotion: {predicted_emotion}')
            
            # Plot probabilities
            st.subheader('Emotion Probabilities')
            plot_probabilities(probabilities)
            
            # Display detailed probabilities
            st.subheader('Detailed Probabilities')
            prob_dict = {emotion: float(prob) for emotion, prob in zip(EMOTIONS, probabilities[0])}
            st.table(prob_dict)
            
            logger.info(f"Predicted emotion: {predicted_emotion}")
    except Exception as e:
        logger.error(f"Error in main app: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main() 