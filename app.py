from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os
from utils.preprocess import extract_url_features, extract_file_features

app = Flask(__name__)

model_dir = "models"

# Load models for web detection
web_models = {
    "SVM": joblib.load(os.path.join(model_dir, "web_svm_model.pkl")),
    "Random Forest": joblib.load(os.path.join(model_dir, "web_rf_model.pkl")),
    "Decision Tree": joblib.load(os.path.join(model_dir, "web_dt_model.pkl")),
    "Logistic Regression": joblib.load(os.path.join(model_dir, "web_lr_model.pkl")),
    "MLP": joblib.load(os.path.join(model_dir, "web_mlp_model.pkl")),
}

# Load models for file detection
file_models = {
    "SVM": joblib.load(os.path.join(model_dir, "file_svm_model.pkl")),
    "Random Forest": joblib.load(os.path.join(model_dir, "file_rf_model.pkl")),
    "Decision Tree": joblib.load(os.path.join(model_dir, "file_dt_model.pkl")),
    "Logistic Regression": joblib.load(os.path.join(model_dir, "file_lr_model.pkl")),
    "MLP": joblib.load(os.path.join(model_dir, "file_mlp_model.pkl")),
}

# Load scalers
scalers = {
    "web": joblib.load(os.path.join(model_dir, "web_scaler.pkl")),
    "file": joblib.load(os.path.join(model_dir, "file_scaler.pkl")),
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict/url", methods=["POST"])
def predict_url():
    url = request.form.get("url")
    if not url:
        return jsonify({"error": "Missing URL"}), 400

    # Extract and scale features
    features = extract_url_features(url)
    df = pd.DataFrame([features])
    X = scalers["web"].transform(df)

    # Predict with all models
    predictions = {}
    votes = 0
    for name, model in web_models.items():
        pred = model.predict(X)[0]
        predictions[name] = int(pred)
        votes += int(pred)

    confidence = round((votes / len(web_models)) * 100, 2)

    return jsonify({
        "input": url,
        "predictions": predictions,
        "confidence": confidence,
        "verdict": "Malware" if confidence >= 50 else "Safe"
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
    votes = 0
    for name, model in file_models.items():
        pred = model.predict(X)[0]
        predictions[name] = int(pred)
        votes += int(pred)

    confidence = round((votes / len(file_models)) * 100, 2)

    return jsonify({
        "input": file.filename,
        "predictions": predictions,
        "confidence": confidence,
        "verdict": "Malware" if confidence >= 50 else "Safe"
    })


if __name__ == "__main__":
    app.run(debug=True)

