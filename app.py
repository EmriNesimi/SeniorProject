from dotenv import load_dotenv
load_dotenv()

import os
import re
import base64
import zipfile

from flask import Flask, render_template, request
import PyPDF2
import google.generativeai as genai

app = Flask(__name__)

# ─── Configure Google Gemini ────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Please set GOOGLE_API_KEY in your .env")
genai.configure(api_key=GOOGLE_API_KEY)
MODEL = genai.GenerativeModel("gemini-1.5-flash")


def classify_with_gemini(prompt: str) -> str:
    """Send a prompt to Gemini and return its response text."""
    resp = MODEL.generate_content(prompt)
    return resp.text.strip() if resp else "Classification failed."


def extract_apk_manifest(fileobj) -> str:
    """
    Open an APK (ZIP) and return the AndroidManifest.xml (UTF-8 decoded
    or base64 if binary). Falls back to entire APK base64.
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

Return one line only: “Real/Legitimate” or “Scam/Fake” (if scam, add a very brief reason).
"""
    return classify_with_gemini(prompt)


def classify_url(url: str) -> str:
    """
    First, apply a set of regex/substring heuristics to catch
    common defacement patterns. If any match, return 'defacement'.
    Otherwise, delegate to Gemini.
    """
    u = url.lower()

    # 1) Joomla-style defacement (option=com_ & view=article)
    if "option=com_" in u and "view=article" in u:
        return "defacement"

    # 2) component template hints
    if "tmpl=" in u or "layout=" in u:
        return "defacement"

    # 3) extremely long Base64 blobs in query (>= 20 chars)
    #    typical of embedded payloads
    qs = u.split("?", 1)[-1]
    if re.search(r"[a-z0-9+/]{20,}={0,2}", qs):
        return "defacement"

    # 4) multiple suspicious params combined
    patterns = ["option=", "link=", "id=", "vsig", "component"]
    if sum(p in u for p in patterns) >= 3:
        return "defacement"

    # fallback to AI
    prompt = f"""
You are a URL security specialist. Classify this URL into exactly one:
- benign
- phishing
- malware
- defacement

URL: {url}

Return exactly one lowercase word.
"""
    return classify_with_gemini(prompt)


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict_url():
    url = request.form.get('url', '').strip()
    if not url:
        return render_template("index.html", url_error="Please enter a URL.")
    # auto-prefix if missing
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "http://" + url

    classification = classify_url(url)
    return render_template("link_result.html",
                           input_url=url,
                           prediction=classification)


@app.route('/scam/', methods=['POST'])
def detect_scam():
    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template("index.html", file_error="Please select a file.")

    f = request.files['file']
    ext = f.filename.lower().rsplit('.', 1)[-1]
    content = ""

    if ext == 'pdf':
        reader = PyPDF2.PdfReader(f)
        pages = [p.extract_text() or "" for p in reader.pages]
        content = "\n".join(pages).strip()
    elif ext == 'txt':
        content = f.read().decode("utf-8", errors="ignore").strip()
    elif ext == 'apk':
        content = extract_apk_manifest(f)
    else:
        return render_template("index.html",
                               file_error="Unsupported file type. Use .apk, .pdf or .txt.")

    if not content:
        return render_template("index.html",
                               file_error="Could not extract any text from file.")

    result = predict_email_scam(content)
    return render_template("file_result.html", prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
