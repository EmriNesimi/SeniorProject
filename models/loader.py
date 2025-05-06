import os
import joblib

BASE = os.path.dirname(__file__)

def load_scaler(source: str):
    """Loads the StandardScaler for either 'app' or 'web'."""
    path = os.path.join(BASE, f"{source}_scaler.joblib")
    return joblib.load(path)

def load_model(source: str, key: str):
    """
    Loads one of the five calibrated classifiers.
    key ∈ {'dt','rf','lr','log','mlp'}
    """
    path = os.path.join(BASE, f"{source}_{key}.joblib")
    return joblib.load(path)
