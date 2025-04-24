import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import joblib

# Paths
data_base   = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
data_path   = os.path.join(data_base, 'data', 'combined_datasets.csv')
models_dir  = os.path.dirname(__file__)

# Load and clean dataset
df = pd.read_csv(data_path, low_memory=False)
df['label'] = df['label'].replace('?', np.nan).pipe(pd.to_numeric, errors='coerce')
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

# Define feature sets
all_features = [c for c in df.columns if c not in ['label', 'source']]
web_features = [
    'web_num_special_chars','web_num_digits','web_has_https','web_has_ip',
    'web_subdomain_count','web_num_params','web_ends_with_number','web_has_suspicious_words'
]

# Clean features
df[all_features] = df[all_features].replace('?', np.nan)
df[all_features] = df[all_features].apply(pd.to_numeric, errors='coerce').fillna(0)

y       = df['label']
sources = df['source'].unique().tolist()
print("Found source labels:", sources)

# Base classifiers
def get_base_models():
    return {
        'dt': DecisionTreeClassifier(max_depth=10, random_state=42),
        'rf': RandomForestClassifier(n_estimators=100, random_state=42),
        'lr': LogisticRegression(solver='lbfgs', max_iter=5000, tol=1e-4, random_state=42),
        'log': LogisticRegression(solver='saga', max_iter=5000, tol=1e-4, random_state=42),
        'mlp': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, tol=1e-4, early_stopping=True, random_state=42)
    }

# Training loop

def train_all():
    base_models = get_base_models()

    for src in sources:
        print(f"\n▶ Training on '{src.upper()}' subset")
        # Select feature matrix
        feat_cols = web_features if src == 'web' else all_features
        X_src = df.loc[df['source'] == src, feat_cols]
        y_src = y.loc[X_src.index]

        # Down-sample web subset
        if src == 'web' and len(X_src) > 20000:
            X_src = X_src.sample(n=20000, random_state=42)
            y_src = y_src.loc[X_src.index]  # re-align labels to the sampled features
            print(f"  Down-sampled web to {len(X_src)} rows")

        # Split train/calibration folds
        X_train, X_calib, y_train, y_calib = train_test_split(
            X_src, y_src,
            test_size=0.2,
            stratify=y_src,
            random_state=42
        )

        # Scale
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_calib_s = scaler.transform(X_calib)

        # Save scaler
        scaler_file = os.path.join(models_dir, f"{src}_scaler.joblib")
        joblib.dump(scaler, scaler_file)
        print(f"  Saved scaler: {os.path.basename(scaler_file)}")

        # Train + calibrate models
        for key, base in base_models.items():
            print(f"  Training & calibrating '{key}'...")
            base.fit(X_train_s, y_train)
            calib = CalibratedClassifierCV(base, method='sigmoid', cv='prefit')
            calib.fit(X_calib_s, y_calib)

            model_file = os.path.join(models_dir, f"{src}_{key}.joblib")
            joblib.dump(calib, model_file)
            print(f"    Saved calibrated model: {os.path.basename(model_file)}")

    print("\nTraining complete.")

if __name__ == '__main__':
    train_all()
