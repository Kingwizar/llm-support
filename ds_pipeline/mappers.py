# ds_pipeline/mappers.py
import re, hashlib, datetime as dt, ast
from typing import Optional, List, Any, Dict
from langdetect import detect, LangDetectException

# ===========================
#        UTIL & PII
# ===========================
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
PHONE_RE = re.compile(r'(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}(?!\d)')
IP_RE    = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')

def _surrogate(s: str, salt="pii"):
    return f"<PII:{hashlib.sha256((salt+s).encode()).hexdigest()[:10]}>"

def anonymize(text: Optional[str]) -> str:
    if not text:
        return ""
    t = EMAIL_RE.sub(lambda m: _surrogate(m.group(), "email"), text)
    t = PHONE_RE.sub(lambda m: _surrogate(m.group(), "phone"), t)
    t = IP_RE.sub(lambda m: _surrogate(m.group(), "ip"), text)
    return t

def detect_lang_safe(text: str) -> Optional[str]:
    try:
        if text and text.strip():
            return detect(text)
    except LangDetectException:
        return None
    return None

def now_iso():
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def stable_row_id(dataset: str, split: str, user_text: Optional[str], agent_text: Optional[str]) -> str:
    base = f"{dataset}|{split}|{user_text or ''}|{agent_text or ''}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]

def clean_scalar(x) -> str:
    """Cast -> str, strip, normalise None/NaN."""
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in ("nan", "none", "null"):
        return ""
    return s

# ===========================
#        SCHÉMA COMMUN
# ===========================
def mk_conv_item(platform: str, dataset: str, split: str, row_id: Any,
                 user_text: Optional[str], agent_text: Optional[str],
                 intent: Optional[str] = None, category: Optional[str] = None,
                 tags: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Construit un document EXACTEMENT conforme au schéma cible.
    """
    u = anonymize(user_text)
    a = anonymize(agent_text)
    lu = detect_lang_safe(u)
    la = detect_lang_safe(a)
    messages = []
    if u:
        messages.append({"role": "user", "text": u, "ts": None, "lang": lu})
    if a:
        messages.append({"role": "agent", "text": a, "ts": None, "lang": la})
    if not messages:
        return {}  # ignore lignes vides

    rid = row_id if row_id not in (None, "", "None") else stable_row_id(dataset, split, u, a)

    return {
        "conversation_id": f"{dataset}:{split}:{rid}",
        "source": {"platform": platform, "dataset": dataset, "split": split},
        "messages": messages,
        "labels": {"intent": intent, "category": category, "resolved": None},
        "meta": {"tags": tags or [], "raw_row_id": row_id, "imported_at": now_iso()},
    }

# ===========================
#        MAPPERS HF
# ===========================

# Talhat/Customer_IT_Support — cols: body, answer, type, queue
def map_hf_talhat(row, split):
    user = row.get("body")
    agent = row.get("answer")
    intent = row.get("type")        # Incident / Request / Problem
    category = row.get("queue")     # Product Support / Billing ...
    rid = row.get("id")             # souvent absent -> fallback hash
    return mk_conv_item("hf", "Talhat/Customer_IT_Support", split, rid, user, agent,
                        intent=intent, category=category, tags=["it","support"])

# VivKatara/customer-support-it-dataset-split — cols: id, body, answer, alternative_body, alternative_answer, dataset_type
def map_hf_vivkatara(row, split):
    user  = row.get("body") or row.get("alternative_body")
    agent = row.get("answer")
    rid   = row.get("id")
    tag   = str(row.get("dataset_type") or "").lower()
    tags  = ["it","support"] + ([tag] if tag else [])
    return mk_conv_item("hf","VivKatara/customer-support-it-dataset-split", split, rid, user, agent,
                        intent=None, category=None, tags=tags)

# harishkotra/technical-support-dataset — cols: text, labels (pas de réponse)
def map_hf_harishkotra(row, split):
    user = row.get("text")
    agent = None
    labels = row.get("labels")
    intent = labels if labels is None else str(labels)
    return mk_conv_item("hf","harishkotra/technical-support-dataset", split, row.get("id"),
                        user, agent, intent=intent, category=None, tags=["technical","support"])

# MakTek/Customer_support_faqs_dataset — cols: question, answer
def map_hf_maktek(row, split):
    return mk_conv_item("hf","MakTek/Customer_support_faqs_dataset", split, row.get("id"),
                        row.get("question"), row.get("answer"),
                        intent="faq", category=None, tags=["faq","generic"])

# balawin/FAQ_Support — col unique: CloudEndure & Successor Services FAQ
def map_hf_balawin(row, split):
    content = row.get("CloudEndure & Successor Services FAQ")
    return mk_conv_item("hf","balawin/FAQ_Support", split, row.get("id"),
                        user_text=None, agent_text=content,
                        intent="faq", category="aws_cloudendure", tags=["faq","generic","aws"])

# ===========================
#     MAPPERS KAGGLE DÉDIÉS
# ===========================

# parthpatil256/it-support-ticket-data — cols: subject, body, answer, type, queue, priority, language, version, tag_1..tag_8
def map_kaggle_parthpatil(row, split):
    subj = clean_scalar(row.get("subject"))
    body = clean_scalar(row.get("body"))
    user_text = (subj + ("\n\n" + body if body else "")).strip() or None
    agent_text = clean_scalar(row.get("answer")) or None

    intent   = clean_scalar(row.get("type")) or None
    category = clean_scalar(row.get("queue")) or None

    tags = ["kaggle","it","support"]
    prio = clean_scalar(row.get("priority"))
    lang = clean_scalar(row.get("language"))
    ver  = clean_scalar(row.get("version"))
    if prio: tags.append(f"priority:{prio}")
    if lang: tags.append(f"lang:{lang}")
    if ver:  tags.append(f"ver:{ver}")
    for i in range(1, 9):
        t = clean_scalar(row.get(f"tag_{i}"))
        if t:
            tags.append(t)

    rid = row.get("id") or row.get("ID") or row.get("Ticket_ID") or getattr(row, "name", None)
    return mk_conv_item("kaggle","parthpatil256/it-support-ticket-data",split,rid,
                        user_text, agent_text, intent=intent, category=category, tags=tags)

# tobiasbueck/multilingual-customer-support-tickets — cols: Unnamed: 0, Body, Department, Priority, Tags (pas de réponse)
def map_kaggle_tobiasbueck(row, split):
    user_text = clean_scalar(row.get("Body")) or None
    agent_text = None
    intent = None
    category = clean_scalar(row.get("Department")) or None
    tags = ["kaggle","it","support"]
    prio = clean_scalar(row.get("Priority"))
    if prio: tags.append(f"priority:{prio}")
    raw_tags = row.get("Tags")
    if raw_tags:
        try:
            for t in ast.literal_eval(str(raw_tags)):
                if t and str(t).strip():
                    tags.append(str(t).strip())
        except Exception:
            tags.append(str(raw_tags))
    rid = row.get("id") or row.get("ID") or row.get("Ticket_ID") or getattr(row, "name", None)
    return mk_conv_item("kaggle","tobiasbueck/multilingual-customer-support-tickets",split,rid,
                        user_text, agent_text, intent=intent, category=category, tags=tags)

# adisongoh/it-service-ticket-classification-dataset — cols: Document, Topic_group (pas de réponse)
def map_kaggle_adisongoh(row, split):
    user_text = clean_scalar(row.get("Document")) or None
    agent_text = None
    intent = clean_scalar(row.get("Topic_group")) or None
    category = intent
    tags = ["kaggle","it","support"]
    rid = row.get("id") or row.get("ID") or row.get("Ticket_ID") or getattr(row, "name", None)
    return mk_conv_item("kaggle","adisongoh/it-service-ticket-classification-dataset",split,rid,
                        user_text, agent_text, intent=intent, category=category, tags=tags)
