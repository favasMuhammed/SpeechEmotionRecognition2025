from setuptools import setup, find_packages

setup(
    name="emotion_detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "librosa",
        "tensorflow",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "sounddevice",
        "soundfile",
        "streamlit",
        "tqdm",
        "onnx",
        "tf2onnx",
        "seaborn"
    ],
    python_requires=">=3.8",
) 