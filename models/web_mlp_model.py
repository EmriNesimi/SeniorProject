# This file represents the trained mlp model for web
# Accuracy: 0.8932
from sklearn.neural_network import MLPClassifier

def get_web_mlp_model():
    return MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
