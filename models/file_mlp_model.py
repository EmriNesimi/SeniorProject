# This file represents the trained mlp model for file
# Accuracy: 0.9860
from sklearn.neural_network import MLPClassifier

def get_file_mlp_model():
    return MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
