from flask import Flask, render_template, request
import pandas as pd
import re

from models.loader import load_scaler, load_model

app = Flask(__name__)

# Load once at startup
file_scaler = load_scaler('app')
file_models = [load_model('app', k) for k in ['dt','rf','lr','log','mlp']]

web_scaler  = load_scaler('web')
web_models  = [load_model('web', k) for k in ['dt','rf','lr','log','mlp']]


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

    # Probability averaging across rows and models
    probas = [m.predict_proba(X)[:, 1] for m in file_models]
    avg_per_row = sum(probas) / len(probas)
    overall_proba = avg_per_row.mean()
    percent = overall_proba * 100

    return render_template('index.html', prediction=f"File Malware Probability: {percent:.1f}%")


@app.route('/predict/url', methods=['POST'])
def predict_url():
    url = request.form.get('url', '')
    if not url:
        return render_template('index.html', prediction="No URL provided.")

    features = {
        'web_num_special_chars': len(re.findall(r'\W', url)),
        'web_num_digits': sum(c.isdigit() for c in url),
        'web_has_https': int('https' in url),
        'web_has_ip': int(bool(re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', url))),
        'web_subdomain_count': max(0, url.count('.') - 1),
        'web_num_params': url.count('='),
        'web_ends_with_number': int(url.rstrip('/').split('/')[-1].isdigit()),
        'web_has_suspicious_words': int(any(w in url.lower() for w in ['login','free','secure','verify','bank']))
    }

    df = pd.DataFrame([features])
    X = web_scaler.transform(df)

    # Average the positive-class probabilities
    probas = [m.predict_proba(X)[0, 1] for m in web_models]
    avg_proba = sum(probas) / len(probas)

    # Use the RF-derived optimal threshold
    threshold = 0.413
    label = "Malicious" if avg_proba >= threshold else "Benign"
    percent = avg_proba * 100

    result = f"URL Malware Probability: {percent:.1f}% ({label})"
    return render_template('index.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)

