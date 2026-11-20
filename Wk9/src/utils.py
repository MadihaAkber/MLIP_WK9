import os
import sys
import joblib

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)

        joblib.dump(obj, file_path)

    except Exception as e:
        raise CustomException(e, sys)