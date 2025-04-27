import os
import re
import json
import logging
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import tldextract
import whois

from sklearn.preprocessing    import StandardScaler
from sklearn.ensemble         import RandomForestClassifier
from sklearn.calibration      import CalibratedClassifierCV
from sklearn.model_selection  import train_test_split
import joblib

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CSV_PATH       = os.path.join(BASE_DIR, 'data', 'links.csv')
MODELS_DIR     = os.path.join(BASE_DIR, 'models')
CACHE_FILENAME = os.path.join(MODELS_DIR, 'domain_age_cache.json')

# ─── Silence WHOIS socket errors ───────────────────────────────────────────────

logging.getLogger('whois').setLevel(logging.CRITICAL)

# ─── Load or initialize domain‐age cache ───────────────────────────────────────

if os.path.exists(CACHE_FILENAME):
    with open(CACHE_FILENAME, 'r') as f:
        _domain_age_cache = json.load(f)
else:
    _domain_age_cache = {}

def _save_cache():
    """Persist the domain‐age cache to disk."""
    with open(CACHE_FILENAME, 'w') as f:
        json.dump(_domain_age_cache, f, indent=2)

# ─── Helper: compute domain age with caching ───────────────────────────────────

def get_domain_age_days(url: str) -> float:
    """
    Return domain age in days since WHOIS creation_date,
    or 0 if unavailable. Caches results to avoid repeated lookups.
    """
    ext    = tldextract.extract(url)
    domain = ext.registered_domain
    if not domain:
        return 0.0

    # Return cached if present
    if domain in _domain_age_cache:
        return _domain_age_cache[domain]

    # Otherwise, attempt WHOIS lookup
    try:
        w  = whois.whois(domain, timeout=5)
        cd = w.creation_date
        # sometimes a list
        if isinstance(cd, list):
            cd = cd[0]
        if isinstance(cd, datetime):
            age = (datetime.now(timezone.utc) - cd).days
        else:
            age = 0.0
    except Exception:
        age = 0.0

    # Cache and return
    _domain_age_cache[domain] = float(age)
    return age

# ─── Load & label dataset ─────────────────────────────────────────────────────

print(f"Loading web URLs from {CSV_PATH}…")
df = pd.read_csv(CSV_PATH)

if 'url' not in df.columns or 'type' not in df.columns:
    raise KeyError("links.csv must contain 'url' and 'type' columns")

# Map all non‐benign types to 1 (malicious), benign→0
df['label'] = (df['type'].str.lower() != 'benign').astype(int)

# ─── Feature engineering ──────────────────────────────────────────────────────

print("Extracting URL features…")
df['web_num_special_chars']    = df['url'].str.count(r'\W')
df['web_num_digits']           = df['url'].str.count(r'\d')
df['web_has_https']            = df['url'].str.startswith('https').astype(int)
df['web_has_ip']               = df['url'].str.contains(
    r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
).astype(int)
df['web_subdomain_count']      = (df['url'].str.count(r'\.') - 1).clip(lower=0)
df['web_num_params']           = df['url'].str.count('=')
df['web_ends_with_number']     = df['url'].str.rstrip('/').str.extract(r'(\d+)$').notnull().astype(int)[0]
df['web_has_suspicious_words'] = df['url'].str.contains(
    r'login|free|secure|verify|bank', flags=re.IGNORECASE
).astype(int)

print("Computing domain ages (cached WHOIS)…")
df['web_domain_age_days'] = df['url'].apply(get_domain_age_days)

# Persist cache so subsequent runs use stored values
_save_cache()

# ─── Prepare arrays for training ──────────────────────────────────────────────

FEATURES = [
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

X = df[FEATURES].fillna(0).to_numpy()
y = df['label'].to_numpy()

X_train, X_calib, y_train, y_calib = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ─── Scale features ───────────────────────────────────────────────────────────

scaler = StandardScaler().fit(X_train)
os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump(scaler, os.path.join(MODELS_DIR, 'web_scaler.joblib'))
X_train_s = scaler.transform(X_train)
X_calib_s = scaler.transform(X_calib)
print("Saved web_scaler.joblib")

# ─── Train & calibrate RandomForest ───────────────────────────────────────────

print("Training RandomForest + sigmoid calibration…")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_s, y_train)

calib = CalibratedClassifierCV(rf, method='sigmoid', cv='prefit')
calib.fit(X_calib_s, y_calib)

joblib.dump(calib, os.path.join(MODELS_DIR, 'web_rf.joblib'))
print("Saved web_rf.joblib")

print("✅ Web model training complete!")

