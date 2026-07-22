import logging
from datetime import datetime
import os
from pathlib import Path

# Get the absolute path of the directory containing this logger.py file
LOG_BASE_DIR = Path(__file__).resolve().parent

# Directory to store logs
LOG_DIR = LOG_BASE_DIR / "rf_logs"

# Timestamped log file
CURRENT_TIME_STAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
LOG_FILE_NAME = f"log_{CURRENT_TIME_STAMP}.log"

# Ensure directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Full log file path
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

# Get the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Create file handler which logs all messages at INFO level
file_handler = logging.FileHandler(LOG_FILE_PATH, mode='w')
file_handler.setLevel(logging.INFO)

# Create console handler which only logs WARNING and above
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)

# Create a common formatter
formatter = logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the root logger
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)