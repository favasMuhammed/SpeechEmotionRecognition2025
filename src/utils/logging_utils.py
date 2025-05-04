"""
Centralized logging utilities for the Speech Emotion Recognition system.
"""

import logging
import os
from datetime import datetime

def setup_logger(name, log_file_prefix):
    """
    Set up a logger with a timestamped log file.

    Args:
        name (str): Name of the logger.
        log_file_prefix (str): Prefix for the log file name (e.g., 'preprocess', 'train').

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    try:
        # Create file handler with timestamp
        log_file = f'logs/{log_file_prefix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger
    except Exception as e:
        # If file handler creation fails, fall back to console-only logging
        logger.addHandler(console_handler)
        logger.error(f"Failed to create file handler: {str(e)}")
        return logger

# Create loggers for different components
preprocess_logger = setup_logger('preprocess', 'preprocess')
train_logger = setup_logger('train', 'train')
app_logger = setup_logger('app', 'app') 