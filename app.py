import os
import base64
import zipfile
from dotenv import load_dotenv

from flask import Flask, render_template, request
import PyPDF2
import google.generativeai as genai

load_dotenv()  # loads GOOGLE_API_KEY from .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Please set GOOGLE_API_KEY in your .env")
genai.configure(api_key=GOOGLE_API_KEY)
MODEL = genai.GenerativeModel("gemini-1.5-flash")

app = Flask(__name__)


def classify_with_gemini(prompt: str) -> str:
    resp = MODEL.generate_content(prompt)
    return resp.text.strip() if resp else "error"


def classify_url_raw(url: str) -> str:
    low = url.lower()
    # Heuristic: Joomla‐style defacement
    if "option=com_" in low or "tmpl=component" in low:
        return "defacement"

    prompt = f"""
You are a URL security specialist. Classify the URL into exactly one of:
- benign
- phishing
- malware
- defacement

URL:
{url}

Return exactly one lowercase word.
"""
    return classify_with_gemini(prompt)


def humanize_url(label: str) -> str:
    m = label.lower()
    return {
        "benign":    "✅ Safe — this link appears non-malicious.",
        "phishing":  "⚠️ Harmful — likely a phishing attempt.",
        "malware":   "❌ Harmful — likely distributes malware.",
        "defacement":"🚩 Defacement — site appears hacked/defaced."
    }.get(m, f"❓ Unknown: {label}")


def extract_apk_manifest(fobj) -> str:
    fobj.seek(0)
    with zipfile.ZipFile(fobj) as z:
        for name in z.namelist():
            if name.lower().endswith("androidmanifest.xml"):
                data = z.read(name)
                try:
                    return data.decode("utf-8", errors="ignore")
                except:
                    return base64.b64encode(data).decode("utf-8")
    fobj.seek(0)
    return base64.b64encode(fobj.read()).decode("utf-8")


def classify_file(fstorage) -> str:
    fn = fstorage.filename.lower()
    content = ""
    if fn.endswith(".pdf"):
        reader = PyPDF2.PdfReader(fstorage)
        content = "\n".join(p.extract_text() or "" for p in reader.pages).strip()
    elif fn.endswith(".txt"):
        content = fstorage.read().decode("utf-8", errors="ignore").strip()
    elif fn.endswith(".apk"):
        content = extract_apk_manifest(fstorage)
    else:
        return "invalid_type"

    if not content:
        return "empty"

    prompt = f"""
You are an expert at spotting scam content in documents.
Analyze the following text and classify it as either:
- Real/Legitimate
- Scam/Fake

Return ONE LINE only: “Real/Legitimate” or “Scam/Fake” (if scam, add brief reason).

Content:
{content}
"""
    return classify_with_gemini(prompt)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict/url', methods=['POST'])
def predict_url():
    raw = request.form.get("url", "").strip()
    if not raw:
        return render_template("link_result.html",
                               url=raw,
                               result_msg="❌ Please enter a URL.",
                               back_url="/")

    # auto-prepend scheme
    url = raw if raw.lower().startswith(("http://","https://")) else f"http://{raw}"
    label = classify_url_raw(url)
    msg   = humanize_url(label)
    return render_template("link_result.html",
                           url=raw,
                           result_msg=msg,
                           back_url="/")


@app.route('/predict/file', methods=['POST'])
def predict_file():
    if 'file' not in request.files:
        msg = "❌ No file uploaded."
    else:
        f = request.files['file']
        if not f.filename:
            msg = "❌ No file selected."
        else:
            verdict = classify_file(f)
            if verdict == "invalid_type":
                msg = "❌ Unsupported file type. Use .pdf, .txt or .apk."
            elif verdict == "empty":
                msg = "⚠️ Could not extract any content."
            else:
                if verdict.lower().startswith("real"):
                    msg = "✅ SAFE — " + verdict
                else:
                    msg = "❌ HARMFUL — " + verdict
    return render_template("file_result.html",
                           result_msg=msg,
                           back_url="/")


if __name__ == "__main__":
    app.run(debug=True)
