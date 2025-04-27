<<<<<<< Updated upstream
from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import re
=======
from dotenv import load_dotenv
load_dotenv()  # pulls GOOGLE_API_KEY into os.environ
>>>>>>> Stashed changes

import os
import base64
import zipfile

from flask import Flask, render_template, request
import PyPDF2
import google.generativeai as genai

app = Flask(__name__)
app.secret_key = "some_secret_key"  # Needed for session

# ─── Configure Gemini ────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Please set GOOGLE_API_KEY in your .env")
genai.configure(api_key=GOOGLE_API_KEY)
MODEL = genai.GenerativeModel("gemini-1.5-flash")

<<<<<<< Updated upstream
# --- Load scalers & models once at startup ---
file_scaler = load_scaler('app')
file_models = [load_model('app', key) for key in ['dt', 'rf', 'lr', 'log', 'mlp']]

web_scaler = load_scaler('web')
web_models = {k: load_model('web', k) for k in ['dt', 'rf', 'lr', 'log', 'mlp']}

# Ensemble / threshold settings
threshold = 0.413
USE_WEIGHTED_ENSEMBLE = True
weights = {'dt': 0.1, 'rf': 0.6, 'lr': 0.1, 'log': 0.1, 'mlp': 0.1}
=======

def classify_with_gemini(prompt: str) -> str:
    resp = MODEL.generate_content(prompt)
    return resp.text.strip() if resp else "Classification failed."


def extract_apk_manifest(fileobj) -> str:
    """
    Open APK (ZIP) and return the AndroidManifest.xml (decoded)
    or else return the entire APK base64‐encoded.
    """
    fileobj.seek(0)
    with zipfile.ZipFile(fileobj) as z:
        for name in z.namelist():
            if name.lower().endswith("androidmanifest.xml"):
                data = z.read(name)
                try:
                    return data.decode("utf-8", errors="ignore")
                except UnicodeDecodeError:
                    return base64.b64encode(data).decode("utf-8")
    # fallback: entire APK
    fileobj.seek(0)
    return base64.b64encode(fileobj.read()).decode("utf-8")


def predict_email_scam(text: str) -> str:
    prompt = f"""
You are an expert at spotting scam emails or text.
Analyze the following content and classify it as either:
- Real/Legitimate
- Scam/Fake

Content:
{text}

Return **one line only**: “Real/Legitimate” or “Scam/Fake” (if scam, add a very brief reason).
"""
    return classify_with_gemini(prompt)


def classify_url(url: str) -> str:
    prompt = f"""
You are a URL security specialist. Classify this URL into exactly one of:
- benign
- phishing
- malware
- defacement

URL: {url}

Return **exactly one** lowercase word.
"""
    return classify_with_gemini(prompt)


# ─── Routes ────────────────────────────────────────────────────────────────────
>>>>>>> Stashed changes

@app.route('/')
def home():
    return render_template("index.html")

<<<<<<< Updated upstream
@app.route('/predict/url', methods=['POST'])
def predict_url():
    url = request.form.get('url', '').strip()
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
        'web_has_suspicious_words': int(any(w in url.lower() for w in ['login', 'free', 'secure', 'verify', 'bank']))
    }
    df = pd.DataFrame([feats])
    X = web_scaler.transform(df[LEGACY_WEB])

    if not USE_WEIGHTED_ENSEMBLE:
        proba = web_models['rf'].predict_proba(X)[0, 1]
    else:
        proba = sum(weights[k] * web_models[k].predict_proba(X)[0, 1] for k in web_models)

    label = "Malicious" if proba >= threshold else "Benign"
    result = f"URL Malware Probability: {proba*100:.1f}% ({label})"
    
    session['prediction'] = result
    return redirect(url_for('link_result'))

@app.route('/predict/file', methods=['POST'])
def predict_file():
    uploaded = request.files.get('file')
    if not uploaded:
        return render_template('index.html', prediction="No file uploaded.")

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        return render_template('index.html', prediction="Error reading file. Please upload a valid CSV.")

    df = df.drop(columns=['source'], errors='ignore')
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    X = file_scaler.transform(df)
    probas = [m.predict_proba(X)[:, 1] for m in file_models]
    avg_per_row = sum(probas) / len(probas)
    overall = avg_per_row.mean()

    label = "Malicious" if overall >= threshold else "Benign"
    result = f"File Malware Probability: {overall*100:.1f}% ({label})"

    session['prediction'] = result
    return redirect(url_for('file_result'))

@app.route('/result/link')
def link_result():
    prediction = session.get('prediction', 'Unknown')
    return render_template('link_result.html', prediction=prediction)

@app.route('/result/file')
def file_result():
    prediction = session.get('prediction', 'Unknown')
    return render_template('file_result.html', prediction=prediction)
=======

@app.route('/predict/url', methods=['POST'])
def predict_url():
    url = request.form.get('url', '').strip()
    context = {}
    if not url.lower().startswith(("http://", "https://")):
        context['url_error'] = "Invalid URL format. Include http:// or https://"
    else:
        classification = classify_url(url)
        context['input_url']      = url
        context['predicted_class'] = classification
    return render_template("index.html", **context)


@app.route('/predict/file', methods=['POST'])
def predict_file():
    context = {}
    if 'file' not in request.files:
        context['file_error'] = "No file part in request."
    else:
        f = request.files['file']
        if not f.filename:
            context['file_error'] = "No file selected."
        else:
            ext = f.filename.lower().rsplit('.', 1)[-1]
            if ext not in ('pdf', 'txt', 'apk'):
                context['file_error'] = "Unsupported file type. Only .pdf, .txt or .apk allowed."
            else:
                # extract text or manifest
                content = ""
                if ext == 'pdf':
                    reader = PyPDF2.PdfReader(f)
                    pages  = [p.extract_text() or "" for p in reader.pages]
                    content = "\n".join(pages)
                elif ext == 'txt':
                    content = f.read().decode('utf-8', errors='ignore')
                else:  # apk
                    content = extract_apk_manifest(f)

                if not content.strip():
                    context['file_error'] = "Could not extract any content from file."
                else:
                    # run the scam detector
                    result = predict_email_scam(content)
                    context['file_result'] = result

    return render_template("index.html", **context)

>>>>>>> Stashed changes

if __name__ == '__main__':
    app.run(debug=True)
