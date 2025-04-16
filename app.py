from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os
from utils.preprocess import extract_url_features, extract_file_features

app = Flask(__name__)

# Load scalers
scalers = {
    "web": joblib.load("models/web_scaler.pkl"),
    "file": joblib.load("models/file_scaler.pkl"),
}

# Import model functions
from models.file_dt import predict as file_dt
from models.file_rf import predict as file_rf
from models.file_lr import predict as file_lr
from models.file_log import predict as file_log
from models.file_mlp import predict as file_mlp

from models.web_dt import predict as web_dt
from models.web_rf import predict as web_rf
from models.web_lr import predict as web_lr
from models.web_log import predict as web_log
from models.web_mlp import predict as web_mlp

file_models = {
    "Decision Tree": file_dt,
    "Random Forest": file_rf,
    "Logistic Regression": file_lr,
    "SVM": file_log,
    "MLP": file_mlp
}

web_models = {
    "Decision Tree": web_dt,
    "Random Forest": web_rf,
    "Logistic Regression": web_lr,
    "SVM": web_log,
    "MLP": web_mlp
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict/url", methods=["POST"])
def predict_url():
    url = request.form.get("url")
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    features = extract_url_features(url)
    df = pd.DataFrame([features])
    X = scalers["web"].transform(df)

    predictions = {}
    vote_count = 0

    for name, model_func in web_models.items():
        pred = model_func(X)
        predictions[name] = int(pred)
        vote_count += int(pred)

    confidence = round((vote_count / 5) * 100, 2)
    verdict = "Malware" if confidence >= 50 else "Safe"

    return jsonify({
        "input": url,
        "predictions": predictions,
        "confidence": confidence,
        "verdict": verdict
    })


@app.route("/predict/file", methods=["POST"])
def predict_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    features = extract_file_features(file)
    df = pd.DataFrame([features])
    X = scalers["file"].transform(df)

    predictions = {}
    vote_count = 0

    for name, model_func in file_models.items():
        pred = model_func(X)
        predictions[name] = int(pred)
        vote_count += int(pred)

    confidence = round((vote_count / 5) * 100, 2)
    verdict = "Malware" if confidence >= 50 else "Safe"

    return jsonify({
        "input": file.filename,
        "predictions": predictions,
        "confidence": confidence,
        "verdict": verdict
    })


if __name__ == "__main__":
    app.run(debug=True)
