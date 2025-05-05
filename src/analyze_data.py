"""
Analyze preprocessed features and labels for Speech Emotion Recognition.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from src.utils.logging_utils import setup_logger

# Set up logger
logger = setup_logger('analyze_data', 'analysis')

def load_processed_data():
    """Load the most recent processed features and labels."""
    processed_dir = Path('data/processed')
    features_dir = processed_dir / 'features'
    labels_dir = processed_dir / 'labels'
    
    # Get most recent files
    feature_files = list(features_dir.glob('features_*.npy'))
    label_files = list(labels_dir.glob('labels_*.npy'))
    
    if not feature_files or not label_files:
        raise FileNotFoundError("No processed data files found")
    
    latest_features = max(feature_files, key=lambda x: x.stat().st_mtime)
    latest_labels = max(label_files, key=lambda x: x.stat().st_mtime)
    
    logger.info(f"Loading features from {latest_features}")
    logger.info(f"Loading labels from {latest_labels}")
    
    features = np.load(latest_features)
    labels = np.load(latest_labels)
    
    return features, labels

def analyze_features(features):
    """Analyze feature statistics."""
    logger.info(f"Feature shape: {features.shape}")
    logger.info(f"Feature mean: {np.mean(features):.3f}")
    logger.info(f"Feature std: {np.std(features):.3f}")
    logger.info(f"Feature min: {np.min(features):.3f}")
    logger.info(f"Feature max: {np.max(features):.3f}")
    
    # Plot feature distributions
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(features.flatten(), bins=50)
    plt.title('Feature Value Distribution')
    plt.xlabel('Feature Value')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=features)
    plt.title('Feature Value Boxplot')
    plt.xlabel('Feature Index')
    plt.ylabel('Value')
    plt.xticks(rotation=90)
    
    plt.tight_layout()
    plt.savefig('data/processed/feature_analysis.png')
    plt.close()

def analyze_labels(labels):
    """Analyze label distribution."""
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    logger.info("Label distribution:")
    for label, count in zip(unique_labels, counts):
        logger.info(f"{label}: {count} samples")
    
    # Plot label distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x=unique_labels, y=counts)
    plt.title('Emotion Distribution')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/processed/label_distribution.png')
    plt.close()

def main():
    """Main analysis function."""
    try:
        # Load data
        features, labels = load_processed_data()
        
        # Analyze features
        logger.info("Analyzing features...")
        analyze_features(features)
        
        # Analyze labels
        logger.info("Analyzing labels...")
        analyze_labels(labels)
        
        logger.info("Analysis complete. Check data/processed/ for visualization files.")
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        raise

if __name__ == '__main__':
    main() 