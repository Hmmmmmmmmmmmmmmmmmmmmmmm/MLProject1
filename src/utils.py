import os
import sys

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import get_logger

import dill

log = get_logger(__name__)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        log.error(f"Error occurred: {e}", exc_info=True)
        raise CustomException(e,sys)