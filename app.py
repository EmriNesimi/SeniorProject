from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os
from utils.preprocess import extract_url_features, extract_file_features

app = Flask(__name__)

# Load trained models and scalers
models = {
    "web": {
        "model": joblib.load("models/web_model.pkl"),
        "scaler": joblib.load("models/web_scaler.pkl")
    },
    "file": {
        "model": joblib.load("models/file_model.pkl"),
        "scaler": joblib.load("models/file_scaler.pkl")
    }
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict/url", methods=["POST"])
def predict_url():
    url = request.form.get("url")
    if not url:
        return jsonify({"error": "Missing URL"}), 400

    try:
        features = extract_url_features(url)
        df = pd.DataFrame([features])
        scaled = models["web"]["scaler"].transform(df)
        prob = models["web"]["model"].predict_proba(scaled)[0][1]
        return jsonify({"malware_probability": round(prob * 100, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict/file", methods=["POST"])
def predict_file():
    if 'file' not in request.files:
        return jsonify({"error": "Missing file"}), 400
    file = request.files['file']
    temp_path = os.path.join("uploads", file.filename)
    file.save(temp_path)

    try:
        features = extract_file_features(temp_path)
        df = pd.DataFrame([features])
        scaled = models["file"]["scaler"].transform(df)
        prob = models["file"]["model"].predict_proba(scaled)[0][1]
        os.remove(temp_path)
        return jsonify({"malware_probability": round(prob * 100, 2)})
    except Exception as e:
        os.remove(temp_path)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
