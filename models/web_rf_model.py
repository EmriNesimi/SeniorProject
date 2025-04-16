# This file represents the trained rf model for web
# Accuracy: 0.8977
from sklearn.ensemble import RandomForestClassifier

def get_web_rf_model():
    return RandomForestClassifier(n_estimators=100, random_state=42)
