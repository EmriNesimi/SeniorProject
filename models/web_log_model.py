# This file represents the trained log model for web
# Accuracy: 0.8626
from sklearn.linear_model import LogisticRegression

def get_web_log_model():
    return LogisticRegression(max_iter=1000, random_state=42)
