import joblib


def load_object(file_path):
    return joblib.load(file_path)
