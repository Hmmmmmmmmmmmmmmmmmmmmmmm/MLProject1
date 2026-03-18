import logging
import os
from datetime import datetime

# LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
# logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE)
# os.makedirs(logs_path,exist_ok=True)

# LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)

# logging.basicConfig(

#     filename = LOG_FILE_PATH,
#     format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
#     level=logging.INFO,

# )


# if __name__ == '__main__':
#     logging.info("Logging Test")

def get_logger(logger_name: str = __name__) -> logging.Logger:
    '''
    Returns a logger configured to log to both file and console.
    Parameters:
        logger_name (str): Name of the logger (usually __name__ of the module)
    Returns:
        logging.Logger: Configured logger instance
    '''
    #create log file if absent
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    #timestamped log file:
    log_file = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
    log_file_path = os.path.join(logs_dir, log_file)
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


if __name__ == "__main__":
    log = get_logger("TestLogger")
    log.info("This is an info message")
    log.warning("This is a warning message")
    log.error("This is an error message")