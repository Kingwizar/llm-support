import re, hashlib, datetime as dt
from langdetect import detect, LangDetectException

EMAIL_RE   = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
PHONE_RE   = re.compile(r'(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}(?!\d)')
IP_RE      = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')

def _surrogate(s: str, salt="pii"):
    return f"<PII:{hashlib.sha256((salt+s).encode()).hexdigest()[:10]}>"

def anonymize(text: str) -> str:
    if not text: return text
    text = EMAIL_RE.sub(lambda m: _surrogate(m.group(), "email"), text)
    text = PHONE_RE.sub(lambda m: _surrogate(m.group(), "phone"), text)
    text = IP_RE.sub(lambda m: _surrogate(m.group(), "ip"), text)
    return text

def detect_lang_safe(text: str):
    try:
        if text and text.strip():
            return detect(text)
    except LangDetectException:
        return None
    return None

def now_iso():
    return dt.datetime.utcnow().replace(microsecond=0).isoformat()+"Z"
