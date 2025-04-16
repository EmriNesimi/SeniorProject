# This file represents the trained rf model for file
# Accuracy: 0.9874
from sklearn.ensemble import RandomForestClassifier

def get_file_rf_model():
    return RandomForestClassifier(n_estimators=100, random_state=42)
