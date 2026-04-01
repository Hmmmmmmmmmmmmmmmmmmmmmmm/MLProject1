import logging
import os
from datetime import datetime

# Global timestamp - created once at module import time
_LOG_TIMESTAMP = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
_LOG_FILE_PATH = None
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _get_log_file_path() -> str:
    '''
    Returns the shared log file path for all loggers.
    Creates logs directory if it doesn't exist.
    '''
    global _LOG_FILE_PATH
    if _LOG_FILE_PATH is None:
        # logs_dir = os.path.join(os.getcwd(), "logs")
        logs_dir = os.path.join(PROJECT_ROOT, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        _LOG_FILE_PATH = os.path.join(logs_dir, f"{_LOG_TIMESTAMP}.log")
    return _LOG_FILE_PATH

def get_logger(logger_name: str = __name__) -> logging.Logger:
    '''
    Returns a logger configured to log to both file and console.
    Uses a shared log file for the entire application session.

    Parameters:
        logger_name (str): Name of the logger (usually __name__ of the module)
    Returns:
        logging.Logger: Configured logger instance
    '''
    log_file_path = _get_log_file_path()
    #Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(name)s | line %(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


# Testing Only:
# if __name__ == "__main__":
#     log = get_logger("TestLogger")
#     log.info("This is an info message")
#     log.warning("This is a warning message")
#     log.error("This is an error message")