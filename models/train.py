import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import os

# === Paths ===
CSV_PATH = "data/combined_datasets.csv"
WEB_SAMPLE_SIZE = 20000

# === Load dataset ===
df = pd.read_csv(CSV_PATH, low_memory=False)
df["source"] = df["source"].astype(str).str.lower().str.strip()

print("Original source counts:")
print(df["source"].value_counts(dropna=False))

# === Split dataset ===
df_file = df[df["source"] == "app"].copy()
df_web = df[df["source"] == "web"].copy()

# === Reduce web rows ===
if len(df_web) > WEB_SAMPLE_SIZE:
    df_web = df_web.sample(n=WEB_SAMPLE_SIZE, random_state=42)
    print(f"Web reduced to {len(df_web)} rows")

# === Model dictionary ===
models = {
    "dt": DecisionTreeClassifier(),
    "rf": RandomForestClassifier(),
    "lr": LinearRegression(),
    "log": LogisticRegression(max_iter=1000),
    "mlp": MLPClassifier(max_iter=500)
}

# === Train function ===
def train_models(df_split, prefix):
    print(f"\nTraining models for: {prefix.upper()}")

    if df_split.empty:
        print(f"⚠️ No data found for {prefix}")
        return

    df_split["label"] = pd.to_numeric(df_split["label"], errors="coerce")
    df_split = df_split[df_split["label"].isin([0, 1])]
    X = df_split.drop(columns=["label", "source"], errors="ignore")
    y = df_split["label"]
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save dummy scaler file
    scaler_path = f"models/{prefix}_scaler.py"
    with open(scaler_path, "w") as f:
        f.write("# This file represents the trained StandardScaler for {}\n".format(prefix))

    # Train each model
    for name, model in models.items():
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            if name == "lr":
                y_pred = np.rint(y_pred)

            acc = accuracy_score(y_test, y_pred)
            print(f"{prefix}_{name} accuracy: {acc:.4f}")

            model_path = f"models/{prefix}_{name}_model.py"
            with open(model_path, "w") as f:
                f.write(f"# This file represents the trained {name} model for {prefix}\n")
                f.write(f"# Accuracy: {acc:.4f}\n")

        except Exception as e:
            print(f"Error training {prefix}_{name}: {e}")

# === Execute training ===
train_models(df_file, "file")
train_models(df_web, "web")
