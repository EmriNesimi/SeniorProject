from dotenv import load_dotenv
load_dotenv()
import os, base64, zipfile
from flask import Flask, render_template, request
import PyPDF2
import google.generativeai as genai

app = Flask(__name__)
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
        for nm in z.namelist():
            if nm.lower().endswith("androidmanifest.xml"):
                data = z.read(nm)
                try:
                    return data.decode("utf-8", errors="ignore")
                except:
                    return base64.b64encode(data).decode()
    fileobj.seek(0)
    return base64.b64encode(fileobj.read()).decode()

def predict_email_scam(text: str) -> str:
    prompt = f"""
You are an expert at spotting scam emails/text.
Analyze the content and classify it as:
- Real/Legitimate
- Scam/Fake

Content:
{text}

Return exactly one line: “Real/Legitimate” or “Scam/Fake” (if scam, add a brief reason).
"""
    return classify_with_gemini(prompt)

def classify_url(url: str) -> str:
    if ".php?option=com_content" in url or ".php?option=com_mailto" in url:
        return "defacement"
    prompt = f"""
You are a URL security specialist. Classify this URL into exactly one of:
benign, phishing, malware, defacement

URL: {url}

Return exactly one lowercase word.
"""
    return classify_with_gemini(prompt).lower()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict/url', methods=['POST'])
def predict_url():
    raw = request.form.get('url', '').strip()
    if not raw:
        return render_template("link_result.html",
                               input_url="",
                               result_type="invalid",
                               detail="No URL provided.",
                               flag="harmful")
    url = raw if raw.startswith(("http://", "https://")) else f"http://{raw}"
    classification = classify_url(url)
    if classification == "benign":
        label, detail, flag = "Safe", "Looks like a normal, harmless site.", "safe"
    elif classification == "phishing":
        label, detail, flag = "Harmful", "Likely a phishing attempt.", "harmful"
    elif classification == "malware":
        label, detail, flag = "Harmful", "Distributes malware.", "harmful"
    else:
        label, detail, flag = "Harmful", "Appears defaced or compromised.", "harmful"
    return render_template("link_result.html",
                           input_url=url,
                           result_type=label,
                           detail=detail,
                           flag=flag)

@app.route('/predict/file', methods=['POST'])
def predict_file():
    f = request.files.get('file')
    if not f or not f.filename:
        return render_template("file_result.html",
                               result_type="Error",
                               detail="No file selected.",
                               flag="harmful")
    ext = f.filename.lower().rsplit('.', 1)[-1]
    if ext not in ('pdf', 'txt', 'apk'):
        return render_template("file_result.html",
                               result_type="Error",
                               detail="Unsupported type. Use PDF, TXT, or APK.",
                               flag="harmful")
    if ext == 'pdf':
        reader = PyPDF2.PdfReader(f)
        text = "\n".join(p.extract_text() or "" for p in reader.pages)
    elif ext == 'txt':
        text = f.read().decode("utf-8", errors="ignore")
    else:
        text = extract_apk_manifest(f)
    if not text.strip():
        return render_template("file_result.html",
                               result_type="Error",
                               detail="Could not extract any text.",
                               flag="harmful")
    verdict = predict_email_scam(text)
    if verdict.lower().startswith("real"):
        label, flag = "Safe — content looks legitimate.", "safe"
    else:
        label, flag = f"Harmful — {verdict}", "harmful"
    return render_template("file_result.html",
                           result_type=label,
                           detail="",
                           flag=flag)

if __name__ == '__main__':
    app.run(debug=True)

