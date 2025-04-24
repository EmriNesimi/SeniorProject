from flask import Flask, render_template, request
import pandas as pd
import re

from models.loader import load_scaler, load_model

app = Flask(__name__)

# --- Legacy web features (no raw URL column) ---
LEGACY_WEB = [
    'web_num_special_chars',
    'web_num_digits',
    'web_has_https',
    'web_has_ip',
    'web_subdomain_count',
    'web_num_params',
    'web_ends_with_number',
    'web_has_suspicious_words'
]

# --- Load scalers & models once at startup ---
file_scaler = load_scaler('app')
file_models = [load_model('app', key) for key in ['dt','rf','lr','log','mlp']]

web_scaler = load_scaler('web')
web_models  = {k: load_model('web', k) for k in ['dt','rf','lr','log','mlp']}

# Ensemble / threshold settings
threshold = 0.413
USE_WEIGHTED_ENSEMBLE = True
weights = {'dt':0.1,'rf':0.6,'lr':0.1,'log':0.1,'mlp':0.1}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/file', methods=['POST'])
def predict_file():
    uploaded = request.files.get('file')
    if not uploaded:
        return render_template('index.html', prediction="No file uploaded.")

    df = pd.read_csv(uploaded)
    df = df.drop(columns=['source'], errors='ignore')
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    X = file_scaler.transform(df)
    probas = [m.predict_proba(X)[:,1] for m in file_models]
    avg_per_row = sum(probas) / len(probas)
    overall = avg_per_row.mean()
    result = f"File Malware Probability: {overall*100:.1f}%"
    return render_template('index.html', prediction=result)

@app.route('/predict/url', methods=['POST'])
def predict_url():
    url = request.form.get('url','').strip()
    if not url:
        return render_template('index.html', prediction="No URL provided.")

    feats = {
        'web_num_special_chars': len(re.findall(r'\W', url)),
        'web_num_digits': sum(c.isdigit() for c in url),
        'web_has_https': int('https' in url),
        'web_has_ip': int(bool(re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', url))),
        'web_subdomain_count': max(0, url.count('.') - 1),
        'web_num_params': url.count('='),
        'web_ends_with_number': int(url.rstrip('/').split('/')[-1].isdigit()),
        'web_has_suspicious_words': int(any(w in url.lower() for w in ['login','free','secure','verify','bank']))
    }
    df = pd.DataFrame([feats])
    X = web_scaler.transform(df[LEGACY_WEB])

    if not USE_WEIGHTED_ENSEMBLE:
        proba = web_models['rf'].predict_proba(X)[0,1]
    else:
        proba = sum(weights[k]*web_models[k].predict_proba(X)[0,1] for k in web_models)
    label = "Malicious" if proba>=threshold else "Benign"
    result = f"URL Malware Probability: {proba*100:.1f}% ({label})"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
