import logging
import os
from datetime import datetime

GLOBAL_LOGGER = None

def setup_logger(ticker: str = None) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    Each run creates a new log file with timestamp and optional ticker.
    
    Args:
        ticker (str, optional): Stock ticker symbol to include in log filename
    
    Returns:
        logging.Logger: Configured logger instance
    """
    global GLOBAL_LOGGER
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Generate log filename with timestamp and ticker
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"hft_strategy_{timestamp}"
    if ticker:
        filename += f"_{ticker}"
    filename += ".log"
    
    log_file = os.path.join(logs_dir, filename)
    
    # Create logger
    logger = logging.getLogger(f"HFTStrategy_{ticker if ticker else 'main'}")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers if any
    logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S.%f'
    )
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    GLOBAL_LOGGER = logger
    return logger
