# Speech Emotion Recognition (SER) System

A real-time speech emotion recognition system that records 5-second audio clips, predicts emotions, and visualizes results using Streamlit. The system supports 9 emotions and is optimized for limited hardware resources.

## Overview

This project implements a Speech Emotion Recognition system that records 5-second audio clips in real-time, predicts emotions, and visualizes the results using Streamlit. The system supports 9 emotions: neutral, calm, happy, sad, angry, fear, disgust, ps (surprised), and boredom. It is optimized for limited hardware (8GB RAM, GTX 1650 with 4GB VRAM) and includes data augmentation, model versioning, data versioning, advanced evaluation metrics, unit tests, and comprehensive validation during preprocessing.

## Features

- Real-time emotion detection from 5-second audio clips
- Support for 9 emotions: neutral, calm, happy, sad, angry, fear, disgust, ps (surprised), boredom
- Data augmentation (pitch shifting, time stretching, noise addition)
- Comprehensive data validation during preprocessing
- Model evaluation with accuracy, precision, recall, F1-score, and confusion matrix
- Model export in multiple formats (.h5, SavedModel, ONNX)
- Model versioning with timestamp-based naming
- Data versioning for preprocessed datasets
- Centralized logging system for preprocessing, training, and inference (INFO level for reduced verbosity)
- Unit tests for audio utilities, preprocessing, training, and app functionality
- Optimized for limited hardware (8GB RAM, 4GB VRAM)

## Project Structure

```
SpeechEmotionRecognition2025/
├── data/                    # Dataset directories
│   ├── RAVDESS/            # RAVDESS dataset
│   └── EMODB/              # EMODB dataset
├── models/                  # Versioned models in .h5, SavedModel, and ONNX formats
├── src/                    # Source code
│   ├── utils/             # Utility modules
│   │   ├── audio_utils.py         # Audio processing (recording, feature extraction)
│   │   ├── augmentation_utils.py   # Audio augmentation functions
│   │   ├── logging_utils.py        # Centralized logging configuration
│   │   └── model_config.py         # Configuration settings
│   ├── app.py             # Streamlit app for real-time emotion detection
│   └── train_model.py     # Model training and evaluation
├── tests/                  # Unit test scripts
│   ├── run_tests.py       # Script to run all tests
│   ├── test_audio_utils.py
│   ├── test_preprocess_data.py
│   ├── test_train_model.py
│   └── test_app.py
├── logs/                   # Directory for log files
├── requirements.txt        # Project dependencies
└── setup.py               # Package setup file
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SpeechEmotionRecognition2025.git
cd SpeechEmotionRecognition2025
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```
Required packages include:
- librosa
- tensorflow-gpu
- numpy
- pandas
- sklearn
- scipy
- matplotlib
- sounddevice
- soundfile
- streamlit
- tqdm
- onnx
- tf2onnx
- seaborn

4. Download datasets:
- RAVDESS: Download from [https://zenodo.org/record/1188976](https://zenodo.org/record/1188976) and place in `data/RAVDESS`
- EMODB: Search for "Berlin Database of Emotional Speech", download, and place in `data/EMODB`

## Usage

1. Preprocess the data:
```bash
python preprocess_data.py
```
This generates versioned preprocessed data (e.g., `data/features_v20250504_120000.npy`) with validation statistics.

2. Train the model:
```bash
python train_model.py
```
This trains the model, evaluates it with multiple metrics, and exports it in .h5, SavedModel, and ONNX formats.

3. Run the Streamlit app:
```bash
streamlit run app.py
```
The app allows selecting model and data versions, and supports real-time emotion detection.

4. Run unit tests:
```bash
python tests/run_tests.py
```
Alternatively, run specific tests:
```bash
python -m unittest tests/test_audio_utils.py
python -m unittest tests/test_preprocess_data.py
python -m unittest tests/test_train_model.py
python -m unittest tests/test_app.py
```

## Module Documentation

### src/utils/audio_utils.py
**Purpose:** Handles audio recording, loading, and feature extraction.
**Key Functions:**
- `record_audio`: Records audio for a specified duration
- `load_audio_file`: Loads audio files with resampling
- `extract_features`: Extracts MFCCs, chroma, and mel-spectrogram features
- `preprocess_audio`: Normalizes audio and extracts features

### src/utils/augmentation_utils.py
**Purpose:** Handles audio augmentation.
**Key Functions:**
- `augment_audio`: Applies pitch shifting, time stretching, and noise addition

### src/utils/logging_utils.py
**Purpose:** Centralizes logging configuration.
**Key Functions:**
- `setup_logger`: Sets up a logger with a timestamped log file

### src/utils/model_config.py
**Purpose:** Defines configuration settings.
**Configurations:**
- `MODEL_CONFIG`: CNN-LSTM architecture settings
- `TRAINING_CONFIG`: Training hyperparameters
- `FEATURE_CONFIG`: Audio feature extraction settings
- `EMOTIONS`: List of supported emotions
- `DATASET_CONFIG`: Dataset processing settings

### preprocess_data.py
**Purpose:** Preprocesses RAVDESS and EMODB datasets with validation and versioning.
**Features:**
- Validates audio files for length, sample rate, silence, and invalid values (NaN/Inf)
- Validates emotion labels for correctness
- Applies data augmentation with validation of augmented audio
- Logs and prints validation statistics
- Saves preprocessed data with version numbers

### train_model.py
**Purpose:** Trains and evaluates the CNN-LSTM model.
**Features:**
- Builds an enhanced model with batch normalization and learning rate scheduling
- Evaluates with accuracy, precision, recall, F1-score, and confusion matrix
- Exports models in .h5, SavedModel, and ONNX formats
- Saves training history for analysis

### app.py
**Purpose:** Runs the Streamlit app for real-time emotion detection.
**Features:**
- Real-time audio recording and prediction
- Supports model and data versioning via dropdown menus
- Visualizes waveform, predicted emotion, probabilities, and detailed table

## Model Architecture

The system uses a CNN-LSTM architecture:
- 3 CNN layers with batch normalization and max-pooling
- 2 LSTM layers with dropout
- 1 Dense layer with dropout
- Output layer with softmax activation
- Learning rate scheduling with exponential decay

## Data Augmentation

The system applies the following augmentations:
- Pitch shifting
- Time stretching
- Noise addition

## Evaluation Metrics

The system provides:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
- Confusion matrix (visualized and saved)

## Hardware Requirements

- CPU: Any modern CPU (for fallback)
- GPU: NVIDIA GTX 1650 (4GB VRAM)
- RAM: 8GB
- Storage: At least 5GB for datasets, models, and logs

## Future Improvements

- Add support for more datasets to expand emotion coverage
- Implement advanced augmentation techniques (e.g., background noise mixing)
- Explore transfer learning with pre-trained models like Wav2Vec 2.0

## License

[Your chosen license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

[Your contact information]
