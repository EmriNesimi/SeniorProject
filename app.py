from flask import Flask, render_template, request
import pandas as pd
import os
import re

from models.file_dt_model import get_file_dt_model
from models.file_rf_model import get_file_rf_model
from models.file_lr_model import get_file_lr_model
from models.file_log_model import get_file_log_model
from models.file_mlp_model import get_file_mlp_model
from models.file_scaler import get_file_scaler

from models.web_dt_model import get_web_dt_model
from models.web_rf_model import get_web_rf_model
from models.web_lr_model import get_web_lr_model
from models.web_log_model import get_web_log_model
from models.web_mlp_model import get_web_mlp_model
from models.web_scaler import get_web_scaler

app = Flask(__name__)

# Load models and scalers
file_dt = get_file_dt_model()
file_rf = get_file_rf_model()
file_lr = get_file_lr_model()
file_log = get_file_log_model()
file_mlp = get_file_mlp_model()
file_scaler = get_file_scaler()

web_dt = get_web_dt_model()
web_rf = get_web_rf_model()
web_lr = get_web_lr_model()
web_log = get_web_log_model()
web_mlp = get_web_mlp_model()
web_scaler = get_web_scaler()

# Dummy fit for web_scaler so .transform won't fail
dummy_url_data = pd.DataFrame([{
    'web_num_special_chars': 5,
    'web_num_digits': 3,
    'web_has_https': 1,
    'web_has_ip': 0,
    'web_subdomain_count': 2,
    'web_num_params': 1,
    'web_ends_with_number': 0,
    'web_has_suspicious_words': 1
}])
web_scaler.fit(dummy_url_data)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/file', methods=['POST'])
def predict_file():
    file = request.files['file']
    if not file:
        return render_template('index.html', prediction="No file uploaded.")

    df = pd.read_csv(file)
    df = df.drop(columns=['source'], errors='ignore')
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    X = file_scaler.fit_transform(df)

    votes = []
    for model in [file_dt, file_rf, file_lr, file_log, file_mlp]:
        prediction = model.predict(X)
        malware_votes = sum(prediction)
        is_malicious = malware_votes > (len(prediction) / 2)
        votes.append(is_malicious)

    malware_percent = (sum(votes) / len(votes)) * 100
    result = f"File Malware Probability: {malware_percent:.0f}%"

    return render_template('index.html', prediction=result)

@app.route('/predict/url', methods=['POST'])
def predict_url():
    url = request.form['url']
    if not url:
        return render_template('index.html', prediction="No URL provided.")

    features = {
        'web_num_special_chars': len(re.findall(r'\W', url)),
        'web_num_digits': sum(c.isdigit() for c in url),
        'web_has_https': int('https' in url),
        'web_has_ip': int(bool(re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', url))),
        'web_subdomain_count': url.count('.') - 1,
        'web_num_params': url.count('='),
        'web_ends_with_number': int(url.rstrip('/').split('/')[-1].isdigit()),
        'web_has_suspicious_words': int(any(x in url.lower() for x in ['login', 'free', 'secure', 'verify', 'bank']))
    }

    df = pd.DataFrame([features])
    X = web_scaler.transform(df)

    votes = []
    for model in [web_dt, web_rf, web_lr, web_log, web_mlp]:
        prediction = model.predict(X)
        is_malicious = int(prediction[0]) == 1
        votes.append(is_malicious)

    malware_percent = (sum(votes) / len(votes)) * 100
    result = f"URL Malware Probability: {malware_percent:.0f}%"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)

