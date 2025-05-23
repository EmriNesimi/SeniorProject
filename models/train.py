import os
import re
import math
import urllib.parse
from datetime import datetime
from collections import Counter

import pandas as pd
import numpy as np
import tldextract
import whois

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

import joblib

def get_domain_age_days(url: str) -> float:
    """Return domain age in days since WHOIS creation_date, or 0 if unavailable."""
    ext = tldextract.extract(url)
    domain = ext.registered_domain
    try:
        w = whois.whois(domain)
        cd = w.creation_date
        if isinstance(cd, list):
            cd = cd[0]
        if not isinstance(cd, datetime):
            return 0.0
        return (datetime.utcnow() - cd).days
    except Exception:
        return 0.0

base_dir   = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
data_path  = os.path.join(base_dir, 'data', 'combined_datasets.csv')
models_dir = os.path.dirname(__file__)

print(f"Loading data from {data_path}...")
df = pd.read_csv(data_path, low_memory=False)

print("Cleaning labels...")
df['label'] = df['label'].replace('?', np.nan)
df['label'] = pd.to_numeric(df['label'], errors='coerce')
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)


if 'url' in df.columns:
    print("URL column found — extracting domain-age feature…")
    df['web_domain_age_days'] = df['url'].apply(get_domain_age_days).fillna(0).astype(float)
else:
    print("No URL column — defaulting domain-age to 0 for all rows")
    df['web_domain_age_days'] = 0.0


print("Defining feature sets…")
file_features = [c for c in df.columns if c.startswith('drebin_')]
web_features = [
    'web_num_special_chars',
    'web_num_digits',
    'web_has_https',
    'web_has_ip',
    'web_subdomain_count',
    'web_num_params',
    'web_ends_with_number',
    'web_has_suspicious_words',
    'web_domain_age_days'
]

missing = [f for f in web_features if f not in df.columns]
if missing:
    raise KeyError(f"Missing web features in dataset: {missing}")

print(f"Using {len(file_features)} file features and {len(web_features)} web features.")

print("Converting feature columns to numeric & filling NaNs…")
df[file_features] = df[file_features].apply(pd.to_numeric, errors='coerce').fillna(0)
df[web_features]  = df[web_features].apply(pd.to_numeric, errors='coerce').fillna(0)


y       = df['label']
sources = df['source'].unique().tolist()
print("Found source labels:", sources)

def get_base_models():
    return {
        'dt':  DecisionTreeClassifier(max_depth=10, random_state=42),
        'rf':  RandomForestClassifier(n_estimators=100, random_state=42),
        'lr':  LogisticRegression(solver='lbfgs',  max_iter=5000, tol=1e-4, random_state=42),
        'log': LogisticRegression(solver='saga',  max_iter=5000, tol=1e-4, random_state=42),
        'mlp': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000,
                              tol=1e-4, early_stopping=True, random_state=42)
    }

def train_all():
    base_models = get_base_models()

    for src in sources:
        print(f"\n▶ Training on '{src.upper()}' subset…")

        if src == 'web':
            X_src = df.loc[df['source'] == src, web_features]
        else:
            X_src = df.loc[df['source'] == src, file_features]
        y_src = y.loc[X_src.index]

        if src == 'web' and len(X_src) > 20000:
            X_src = X_src.sample(n=20000, random_state=42)
            y_src = y_src.loc[X_src.index]
            print(f"  Down-sampled web to {len(X_src)} rows")

        X_train, X_calib, y_train, y_calib = train_test_split(
            X_src, y_src, test_size=0.2, stratify=y_src, random_state=42
        )

        # Fit scaler
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_calib_s = scaler.transform(X_calib)

        scaler_path = os.path.join(models_dir, f"{src}_scaler.joblib")
        joblib.dump(scaler, scaler_path)
        print(f"  Saved scaler: {os.path.basename(scaler_path)}")

        # Train & calibrate each model
        for key, base in base_models.items():
            print(f"  Training & calibrating '{key}'…")
            base.fit(X_train_s, y_train)
            calib = CalibratedClassifierCV(base, method='sigmoid', cv='prefit')
            calib.fit(X_calib_s, y_calib)

            model_path = os.path.join(models_dir, f"{src}_{key}.joblib")
            joblib.dump(calib, model_path)
            print(f"    Saved calibrated model: {os.path.basename(model_path)}")

    print("\nAll training and calibration complete!")

if __name__ == '__main__':
    train_all()
