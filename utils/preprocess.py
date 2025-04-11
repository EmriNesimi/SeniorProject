import pandas as pd
from urllib.parse import urlparse
import numpy as np
import io

# === URL Feature Extraction ===
def extract_url_features(url):
    parsed = urlparse(url)
    domain = parsed.netloc

    def entropy(s):
        prob = [float(s.count(c)) / len(s) for c in set(s)]
        return -sum([p * np.log2(p) for p in prob])

    suspicious_words = ['login', 'verify', 'secure', 'account']
    shortening_services = ['bit.ly', 'goo.gl', 'tinyurl.com', 'ow.ly']

    features = {
        "web_url_length": len(url),
        "web_num_special_chars": sum(1 for c in url if c in "_-?&=/."),
        "web_num_digits": sum(1 for c in url if c.isdigit()),
        "web_has_https": int("https" in url.lower()),
        "web_has_ip": int(any(c.isdigit() for c in domain)),
        "web_subdomain_count": domain.count('.') - 1,
        "web_num_params": url.count('='),
        "web_ends_with_number": int(url.strip('/').split('/')[-1].isdigit()),
        "web_has_suspicious_words": int(any(w in url.lower() for w in suspicious_words)),
        "web_contains_dash": int('-' in url),
        "web_contains_at_symbol": int('@' in url),
        "web_url_entropy": round(entropy(url), 2),
        "web_contains_double_slash": int('//' in url[7:]),
        "web_contains_https_token": int('https' in url.lower() and 'https://' not in url.lower()),
        "web_url_depth": url.count('/'),
        "web_has_port": int(':' in url[7:]),
        "web_has_email_like": int('.@' in url or '@.' in url or url.count('@') > 1),
        "web_is_shortened": int(any(s in url.lower() for s in shortening_services)),
        "web_contains_equal": int('=' in url),
        "web_contains_plus": int('+' in url),
    }

    return features


# === File Feature Extraction (from CSV) ===
def extract_file_features(file):
    try:
        df = pd.read_csv(file)
        df_numeric = df.select_dtypes(include=['number'])
        features = df_numeric.mean().to_dict()
        return features
    except Exception as e:
        print(f"[Error] Couldn't parse uploaded file: {e}")
        return {}
