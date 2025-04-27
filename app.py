from dotenv import load_dotenv
load_dotenv()  # pull GOOGLE_API_KEY into os.environ

import os
import base64
import zipfile

from flask import Flask, render_template, request
import PyPDF2
import google.generativeai as genai

app = Flask(__name__)

# ─── Configure Gemini ────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Please set GOOGLE_API_KEY in your .env")
genai.configure(api_key=GOOGLE_API_KEY)
MODEL = genai.GenerativeModel("gemini-1.5-flash")


def classify_with_gemini(prompt: str) -> str:
    resp = MODEL.generate_content(prompt)
    return resp.text.strip() if resp else "Classification failed."


def extract_apk_manifest(fileobj) -> str:
    fileobj.seek(0)
    with zipfile.ZipFile(fileobj) as z:
        for name in z.namelist():
            if name.lower().endswith("androidmanifest.xml"):
                data = z.read(name)
                try:
                    return data.decode("utf-8", errors="ignore")
                except UnicodeDecodeError:
                    return base64.b64encode(data).decode("utf-8")
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

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict/url', methods=['POST'])
def predict_url():
    # grab the raw user input
    raw = request.form.get('url', '').strip()
    if not raw:
        return render_template("index.html", url_error="Please enter a URL.")

    # if no scheme, prepend http://
    if not raw.lower().startswith(("http://", "https://")):
        scan_url = "http://" + raw
    else:
        scan_url = raw

    classification = classify_url(scan_url)

    # render your link_result.html, passing both the original & the classification
    return render_template(
        "link_result.html",
        original_url=raw,
        prediction=classification
    )


@app.route('/predict/file', methods=['POST'])
def predict_file():
    if 'file' not in request.files:
        return render_template("index.html", file_error="No file uploaded.")
    f = request.files['file']
    if not f.filename:
        return render_template("index.html", file_error="No file selected.")

    ext = f.filename.lower().rsplit('.', 1)[-1]
    if ext not in ('pdf', 'txt', 'apk'):
        return render_template("index.html", file_error="Unsupported file type.")

    if ext == 'pdf':
        reader = PyPDF2.PdfReader(f)
        content = "\n".join([p.extract_text() or "" for p in reader.pages]).strip()
    elif ext == 'txt':
        content = f.read().decode("utf-8", errors="ignore").strip()
    else:
        content = extract_apk_manifest(f)

    if not content:
        return render_template("index.html", file_error="Couldn’t extract any text from the file.")

    result = predict_email_scam(content)
    return render_template("file_result.html", prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
