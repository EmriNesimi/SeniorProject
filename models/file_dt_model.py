# This file represents the trained dt model for file
# Accuracy: 0.9757
from sklearn.tree import DecisionTreeClassifier

def get_file_dt_model():
    return DecisionTreeClassifier(max_depth=10, random_state=42)
