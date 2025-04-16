# This file represents the trained dt model for web
# Accuracy: 0.8911
from sklearn.tree import DecisionTreeClassifier

def get_web_dt_model():
    return DecisionTreeClassifier(max_depth=10, random_state=42)
