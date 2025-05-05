# Speech Emotion Recognition System

A real-time speech emotion recognition system that uses deep learning to classify emotions from audio input. The system processes audio from both the RAVDESS and EMODB datasets and can perform real-time emotion detection through a user-friendly Streamlit interface.

## Features

- **Real-time Emotion Detection**: Record and analyze speech in real-time
- **Multi-Dataset Support**: Process both RAVDESS and EMODB datasets
- **Data Augmentation**: Enhance training data with pitch shifting, time stretching, and noise addition
- **Advanced Model Architecture**: CNN-LSTM hybrid model for improved accuracy
- **Comprehensive Validation**: Extensive input validation and error handling
- **User-Friendly Interface**: Streamlit-based web interface with visualizations
- **Model Export**: Support for multiple model formats (Keras, ONNX)

## Project Structure

```
SpeechEmotionRecognition2025/
├── data/                  # Dataset and preprocessed data
│   ├── RAVDESS/          # RAVDESS dataset
│   ├── EMODB/            # EMODB dataset
│   └── features_*.npy    # Preprocessed features
├── models/               # Trained models
├── src/                  # Source code
│   ├── app.py           # Streamlit application
│   ├── preprocess_data.py # Data preprocessing
│   ├── train_model.py   # Model training
│   └── utils/           # Utility functions
│       ├── audio_utils.py    # Audio processing
│       ├── model_config.py   # Model configuration
│       └── logging_utils.py  # Logging utilities
├── tests/               # Test files
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Recent Improvements

### 1. Enhanced Error Handling and Validation
- Added comprehensive input validation for all functions
- Improved error messages and logging
- Added type checking for function parameters
- Implemented shape validation for features and model inputs/outputs

### 2. Audio Processing Improvements
- Centralized audio augmentation in `audio_utils.py`
- Added validation for audio signals and features
- Improved audio recording functionality
- Enhanced feature extraction with proper error handling

### 3. Model Training Enhancements
- Added model architecture validation
- Improved training process with better error handling
- Added support for model export in multiple formats
- Enhanced model configuration management

### 4. Application Improvements
- Added better user feedback in the Streamlit interface
- Improved data and model version management
- Enhanced visualization of results
- Added detailed probability display

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SpeechEmotionRecognition2025.git
cd SpeechEmotionRecognition2025
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
..\venv\Scripts\activate
# On Unix/MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download datasets:
- RAVDESS: https://zenodo.org/record/1188976
- EMODB: Search for 'Berlin Database of Emotional Speech'

5. Place datasets in the `data` directory:
```
data/
├── RAVDESS/
└── EMODB/
```

## Usage

1. Preprocess the datasets:
```bash
python src/preprocess_data.py
```

2. Train the model:
```bash
python src/train_model.py
```

3. Run the Streamlit app:
```bash
streamlit run src/app.py
```

## Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

## Configuration

The system can be configured through the following files:
- `src/utils/model_config.py`: Model architecture and training parameters
- `src/utils/audio_utils.py`: Audio processing parameters
- `src/utils/logging_utils.py`: Logging configuration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- RAVDESS dataset creators
- EMODB dataset creators
- TensorFlow and Streamlit communities
