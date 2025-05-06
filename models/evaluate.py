import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from loader import load_scaler, load_model

WEB_FEATURES = [
    'web_num_special_chars',
    'web_num_digits',
    'web_has_https',
    'web_has_ip',
    'web_subdomain_count',
    'web_num_params',
    'web_ends_with_number',
    'web_has_suspicious_words'
]

def load_and_clean_data(data_path):
    """
    Load CSV, clean labels and features, return DataFrame.
    """
    df = pd.read_csv(data_path, low_memory=False)
    # Clean label
    df['label'] = df['label'].replace('?', np.nan)
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    # Clean URL features
    df[WEB_FEATURES] = df[WEB_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
    return df


def evaluate_source(src, df, feat_cols, threshold=0.5, plot_roc=False):
    """
    Evaluate models for a given source ('app' or 'web').
    Prints ROC AUC and classification report for each model.
    Optionally plots ROC curve for the best model.
    """
    sub = df[df.source == src]
    X = sub[feat_cols]
    y = sub['label']

    scaler = load_scaler(src)
    Xs = scaler.transform(X)

    models = {key: load_model(src, key) for key in ['dt','rf','lr','log','mlp']}

    print(f"\n=== EVALUATION FOR '{src.upper()}' SUBSET ===")
    best_auc = 0
    best_key = None
    best_probas = None

    for key, m in models.items():
        proba = m.predict_proba(Xs)[:,1]
        preds = (proba >= threshold).astype(int)
        auc = roc_auc_score(y, proba)
        print(f"\n--- Model: {key} ---")
        print(f"ROC AUC: {auc:.4f}")
        print(classification_report(y, preds))
        if auc > best_auc:
            best_auc = auc
            best_key = key
            best_probas = proba

    print(f"\nBest model: {best_key} with ROC AUC = {best_auc:.4f}")

    fpr, tpr, thresholds = roc_curve(y, best_probas)
    j_scores = tpr - fpr
    j_idx = np.argmax(j_scores)
    optimal_thresh = thresholds[j_idx]
    print(f"Optimal threshold for {best_key}: {optimal_thresh:.3f} (TPR={tpr[j_idx]:.3f}, FPR={fpr[j_idx]:.3f})")

    if plot_roc:
        plt.figure()
        plt.plot(fpr, tpr, label=f"{best_key} (AUC={best_auc:.2f})")
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve — {src.upper()}::{best_key}")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'combined_datasets.csv'))
utils_df = load_and_clean_data(data_path)

evaluate_source('web', utils_df, WEB_FEATURES, threshold=0.5, plot_roc=True)