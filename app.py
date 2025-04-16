from flask import Flask, render_template, request
import os
import pandas as pd

from models.file_dt_model import model as file_dt
from models.file_rf_model import model as file_rf
from models.file_lr_model import model as file_lr
from models.file_log_model import model as file_log
from models.file_mlp_model import model as file_mlp
from models.file_scaler import scaler as file_scaler

from models.web_dt_model import model as web_dt
from models.web_rf_model import model as web_rf
from models.web_lr_model import model as web_lr
from models.web_log_model import model as web_log
from models.web_mlp_model import model as web_mlp
from models.web_scaler import scaler as web_scaler

app = Flask(__name__)

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
    X = file_scaler.transform(df)

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

    import re
    features = {
        'web_num_special_chars': len(re.findall(r'\W', url)),
        'web_num_digits': sum(c.isdigit() for c in url),
        'web_has_https': int('https' in url),
        'web_has_ip': int(bool(re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', url))),
        'web_subdomain_count': url.count('.') - 1,
        'web_num_params': url.count('='),
        'web_ends_with_number': int(url.rstrip('/').split('/')[-1].isdigit()),
        'web_has_suspicious_words': int(any(x in url for x in ['login', 'free', 'secure', 'verify', 'bank']))
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
