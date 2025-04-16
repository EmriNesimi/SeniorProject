# This file represents the trained log model for file
# Accuracy: 0.9781
from sklearn.linear_model import LogisticRegression

def get_file_log_model():
    return LogisticRegression(max_iter=1000, random_state=42)
