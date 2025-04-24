import os
import joblib

def load_scaler(source: str):
    """
    Load a pre-fitted StandardScaler for the given data source.

    Parameters
    ----------
    source : str
        Identifier matching the scaler filename (e.g. 'app' or 'web').

    Returns
    -------
    sklearn.preprocessing.StandardScaler
    """
    filename = f"{source}_scaler.joblib"
    path = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler file not found: {path}")
    return joblib.load(path)


def load_model(source: str, key: str):
    """
    Load a calibrated classifier for the given source and model key.

    Parameters
    ----------
    source : str
        Identifier matching the model group (e.g. 'app' or 'web').
    key : str
        Short code for which model to load (one of 'dt', 'rf', 'lr', 'log', 'mlp').

    Returns
    -------
    sklearn.base.BaseEstimator
        A fitted and calibrated scikit-learn estimator.
    """
    filename = f"{source}_{key}.joblib"
    path = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)
