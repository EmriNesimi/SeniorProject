import os
import re
import math
from urllib.parse import urlparse

# === URL Feature Extraction ===
def extract_url_features(url):
    parsed = urlparse(url)
    netloc = parsed.netloc
    path = parsed.path
    query = parsed.query

    features = {
        'url_length': len(url),
        'num_special_chars': sum(c in url for c in '-_?&=/.%'),
        'num_digits': sum(c.isdigit() for c in url),
        'has_https': int(parsed.scheme == 'https'),
        'has_ip': int(re.match(r"^(\d{1,3}\.){3}\d{1,3}$", netloc) is not None),
        'subdomain_count': netloc.count('.') - 1,
        'num_params': query.count('='),
        'ends_with_number': int(path.strip('/').split('/')[-1].isdigit()),
        'has_suspicious_words': int(any(w in url.lower() for w in ['login', 'secure', 'account', 'verify'])),
    }
    return features

# === File Feature Extraction (Dummy Example) ===
def extract_file_features(file_path):
    # NOTE: Replace this with real static/dynamic file analysis logic
    file_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f:
        byte_data = f.read(1024)
        entropy = -sum(byte_data.count(b) / len(byte_data) * math.log2(byte_data.count(b) / len(byte_data))
                       for b in set(byte_data) if byte_data.count(b) != 0)

    features = {
        'file_size': file_size,
        'file_entropy': entropy,
        'has_mz_header': int(open(file_path, 'rb').read(2) == b'MZ')  # Basic PE header check (Windows files)
    }
    return features