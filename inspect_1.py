#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import joblib
import tldextract
import whois
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler

DATA_CSV = os.path.join(os.path.dirname(__file__), "data", "combined_datasets.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

FILE_PREFIXES = ["app", "web"]
MODEL_KEYS    = ["dt", "rf", "lr", "log", "mlp"]

WEB_FEATURES = [
    "web_num_special_chars",
    "web_num_digits",
    "web_has_https",
    "web_has_ip",
    "web_subdomain_count",
    "web_num_params",
    "web_ends_with_number",
    "web_has_suspicious_words",
    "web_domain_age_days",
]

def get_domain_age_days(url: str) -> float:
    ext = tldextract.extract(url)
    dom = ext.registered_domain
    try:
        w = whois.whois(dom)
        cd = w.creation_date
        if isinstance(cd, list): cd = cd[0]
        if not isinstance(cd, datetime): return 0.0
        return (datetime.utcnow() - cd).days
    except Exception:
        return 0.0

def load_and_prepare():
    # 1) load
    df = pd.read_csv(DATA_CSV, low_memory=False)
    # 2) clean label
    df["label"] = pd.to_numeric(df["label"].replace("?", np.nan), errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    # 3) domain age
    if "url" in df.columns:
        df["web_domain_age_days"] = df["url"].apply(get_domain_age_days)
    else:
        df["web_domain_age_days"] = 0.0
    # 4) numeric coercion
    file_feats = [c for c in df.columns if c.startswith("drebin_")]
    df[file_feats] = df[file_feats].apply(pd.to_numeric, errors="coerce").fillna(0)
    df[WEB_FEATURES] = df[WEB_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    return df, file_feats

def evaluate():
    df, file_feats = load_and_prepare()
    y = df["label"]

    for src in FILE_PREFIXES:
        print(f"\n=== Evaluating source: {src.upper()} ===")
        if src == "web":
            X = df[df["source"] == src][WEB_FEATURES]
        else:
            X = df[df["source"] == src][file_feats]
        y_true = y.loc[X.index]

        # load scaler
        scaler_path = os.path.join(MODELS_DIR, f"{src}_scaler.joblib")
        scaler = joblib.load(scaler_path)

        X_scaled = scaler.transform(X)

        for key in MODEL_KEYS:
            model_path = os.path.join(MODELS_DIR, f"{src}_{key}.joblib")
            if not os.path.exists(model_path):
                print(f"  → missing {model_path}, skipping")
                continue

            clf = joblib.load(model_path)

            # some calibrated models might not implement predict_proba
            try:
                proba = clf.predict_proba(X_scaled)[:, 1]
                rocauc = roc_auc_score(y_true, proba)
            except Exception:
                rocauc = float("nan")

            y_pred = clf.predict(X_scaled)
            acc    = accuracy_score(y_true, y_pred)
            prec   = precision_score(y_true, y_pred, zero_division=0)
            rec    = recall_score(y_true, y_pred, zero_division=0)
            f1     = f1_score(y_true, y_pred, zero_division=0)

            print(f"\n  Model: {key}")
            print(f"    ROC AUC:    {rocauc:.4f}")
            print(f"    Accuracy:   {acc:.4f}")
            print(f"    Precision:  {prec:.4f}")
            print(f"    Recall:     {rec:.4f}")
            print(f"    F1 Score:   {f1:.4f}")
            print("    Classification report:")
            print(classification_report(y_true, y_pred, digits=4))

if __name__ == "__main__":
    evaluate()
