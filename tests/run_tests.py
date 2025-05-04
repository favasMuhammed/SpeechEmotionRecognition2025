import unittest
import sys
import os
import logging
import subprocess
from datetime import datetime
import numpy as np

def setup_logging():
    """Set up logging configuration."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def activate_venv():
    """Activate the virtual environment."""
    venv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'venv')
    if sys.platform == 'win32':
        activate_script = os.path.join(venv_path, 'Scripts', 'activate.bat')
        subprocess.run([activate_script], shell=True)
    else:
        activate_script = os.path.join(venv_path, 'bin', 'activate')
        subprocess.run(['source', activate_script], shell=True)

def setup_test_environment():
    """Set up the test environment."""
    # Ensure we're in the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # Create necessary directories
    directories = [
        "tests/data",
        "tests/models",
        "logs",
        "data/RAVDESS",
        "data/EMODB"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")
    
    # Check for dataset availability
    from src.utils.model_config import DATASET_CONFIG
    missing_datasets = []
    
    if not os.path.exists(DATASET_CONFIG['ravdess_path']) or not os.listdir(DATASET_CONFIG['ravdess_path']):
        missing_datasets.append("RAVDESS")
        logging.warning("RAVDESS dataset not found or empty")
    
    if not os.path.exists(DATASET_CONFIG['emodb_path']) or not os.listdir(DATASET_CONFIG['emodb_path']):
        missing_datasets.append("EMODB")
        logging.warning("EMODB dataset not found or empty")
    
    if missing_datasets:
        logging.warning(f"Missing datasets: {', '.join(missing_datasets)}")
        logging.warning("Some tests may be skipped due to missing datasets")
        
        # Create a mock dataset for testing
        logging.info("Creating mock dataset for testing")
        mock_audio = np.zeros(44100 * 5)  # 5 seconds of silence
        for dataset in missing_datasets:
            dataset_path = DATASET_CONFIG[f"{dataset.lower()}_path"]
            for i in range(10):  # Create 10 mock files
                file_path = os.path.join(dataset_path, f"mock_{i}.wav")
                import soundfile as sf
                sf.write(file_path, mock_audio, 44100)
                logging.info(f"Created mock audio file: {file_path}")

def run_test_suite():
    """Run the test suite and return the result."""
    # Discover all test files
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Log test results
    logging.info(f"Tests run: {result.testsRun}")
    logging.info(f"Failures: {len(result.failures)}")
    logging.info(f"Errors: {len(result.errors)}")
    logging.info(f"Skipped: {len(result.skipped)}")
    
    return result

def cleanup_test_environment():
    """Clean up test environment after running tests."""
    # Remove test directories
    test_dirs = [
        "tests/data",
        "tests/models"
    ]
    for directory in test_dirs:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(directory)
            logging.info(f"Removed directory: {directory}")

def main():
    """Main function to run all tests."""
    try:
        # Set up logging
        logger = setup_logging()
        logger.info("Starting test run")
        
        # Activate virtual environment
        logger.info("Activating virtual environment")
        activate_venv()
        
        # Set up test environment
        logger.info("Setting up test environment")
        setup_test_environment()
        
        # Run tests
        logger.info("Running test suite")
        result = run_test_suite()
        
        # Clean up
        logger.info("Cleaning up test environment")
        cleanup_test_environment()
        
        # Exit with appropriate status code
        exit_code = 0 if result.wasSuccessful() else 1
        logger.info(f"Test run completed with exit code: {exit_code}")
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"Error during test run: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 