from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from models.file_scaler import get_file_scaler
from models.web_scaler import get_web_scaler
from models.file_dt_model import get_file_dt_model
from models.file_rf_model import get_file_rf_model
from models.file_lr_model import get_file_lr_model
from models.file_log_model import get_file_log_model
from models.file_mlp_model import get_file_mlp_model
from models.web_dt_model import get_web_dt_model
from models.web_rf_model import get_web_rf_model
from models.web_lr_model import get_web_lr_model
from models.web_log_model import get_web_log_model
from models.web_mlp_model import get_web_mlp_model

app = Flask(__name__)

# === Load Models ===
file_models = {
    "Decision Tree": get_file_dt_model(),
    "Random Forest": get_file_rf_model(),
    "Linear Regression": get_file_lr_model(),
    "Logistic Regression": get_file_log_model(),
    "MLP Classifier": get_file_mlp_model()
}

web_models = {
    "Decision Tree": get_web_dt_model(),
    "Random Forest": get_web_rf_model(),
    "Linear Regression": get_web_lr_model(),
    "Logistic Regression": get_web_log_model(),
    "MLP Classifier": get_web_mlp_model()
}

# === Load Scalers ===
file_scaler = get_file_scaler()
web_scaler = get_web_scaler()

def predict_malware(input_data, source_type):
    if source_type == "file":
        models = file_models
        scaler = file_scaler
    elif source_type == "web":
        models = web_models
        scaler = web_scaler
    else:
        return "Unknown source type"

    df = pd.DataFrame([input_data])
    X_scaled = scaler.transform(df)
    
    malware_votes = 0
    total_models = len(models)

    for name, model in models.items():
        prediction = model.predict(X_scaled)
        if name == "Linear Regression":
            prediction = np.round(prediction)
        if prediction[0] == 1:
            malware_votes += 1

    confidence = (malware_votes / total_models) * 100
    return confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        source_type = request.form['source_type']
        features = request.form['features']
        try:
            input_data = eval(features)
            prediction = predict_malware(input_data, source_type)
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
