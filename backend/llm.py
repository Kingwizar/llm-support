# rag_core_only.py
# -*- coding: utf-8 -*-
"""
RAG 'core' (sans modèle LLM) :
- Télécharge les documents KB depuis Object Storage via PAR READ
- Découpe en chunks -> Embeddings (Sentence-Transformers) -> Index FAISS
- Retrieval Top-K avec déduplication
- Construit un PROMPT complet à envoyer à un modèle (endpoint, local, etc.)
- Nettoie la réponse brute du modèle (clean_llm_answer)
"""

import os, re, json, hashlib
from typing import List, Dict, Any
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


# ===================== CONFIG =====================

PAR_READ_BASE = "https://objectstorage.eu-frankfurt-1.oraclecloud.com/p/frLlNzlzPMugbTD_QCtbnlWMROwmubk5t5vhwdN0QZEQefnp7nhUo3EgYwiK2O03/n/frleg9qvtz9u/b/llm-support-data/o/kb/"

DOC_OBJECTS = [
    "networking_rdp.md",
    "sla_v3_2025-08-20.json",
    "kb-442.jsonl",
]

INDEX_DIR = "rag_index"
os.makedirs(INDEX_DIR, exist_ok=True)

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K_DEFAULT = 4
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ===================== SYSTEM & PROMPT =====================

# ===================== SYSTEM & PROMPT =====================

SYSTEM_RAG = """# System: N+One Datacenter Customer Support (RAG)
You are N+One Datacenter Customer Support Assistant.
...
(End of system)
"""

RUNTIME_TEMPLATE = """[SYSTEM]
{system}

[USER QUESTION]
{question}

[RETRIEVED CONTEXT]
{sources_block}

[RESPONSE REQUIREMENTS]
- Start with Summary
- Then Procedure
- Add Citations
- Add Assumptions
- Ask clarifying questions if needed
"""



# ===================== FETCH & CHUNK =====================

def _par_url(object_name: str) -> str:
    """Construit l’URL complète d’un objet dans le bucket."""
    return PAR_READ_BASE.rstrip("/") + "/" + object_name.lstrip("/")

def fetch_object_text(object_name: str) -> str:
    """Télécharge le contenu brut d’un objet (texte)."""
    url = _par_url(object_name)
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    return r.text

def read_text_from_object(object_name: str) -> str:
    """Lit et normalise le texte d’un objet (jsonl, json, csv, md, txt)."""
    txt = fetch_object_text(object_name)
    name = object_name.lower()

    if name.endswith(".jsonl"):
        lines = []
        for line in txt.splitlines():
            try:
                obj = json.loads(line)
                lines.append(obj.get("text") or obj.get("content") or line)
            except:
                lines.append(line)
        return "\n".join(lines)

    if name.endswith(".json"):
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict):
                return obj.get("text") or obj.get("content") or json.dumps(obj)
            if isinstance(obj, list):
                return "\n".join(
                    [(o.get("text") or o.get("content") or json.dumps(o)) if isinstance(o, dict) else str(o)
                     for o in obj]
                )
        except:
            return txt

    if name.endswith(".csv"):
        try:
            import pandas as pd
            from io import StringIO
            df = pd.read_csv(StringIO(txt))
            col = df.columns[0]
            return "\n".join(df[col].astype(str).tolist())
        except:
            return txt

    return txt

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Découpe un texte en morceaux avec chevauchement."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    chunks, i = [], 0
    step = size - overlap
    while i < len(text):
        chunks.append(text[i:i + size])
        i += step
    return chunks

# ===================== INDEX (FAISS) =====================

def build_index_from_bucket(objects: List[str]) -> None:
    """Construit un index FAISS à partir des documents du bucket."""
    records = []
    for obj in objects:
        raw = read_text_from_object(obj)
        for ch in chunk_text(raw):
            records.append({"doc": obj, "text": ch})

    if not records:
        raise RuntimeError("Aucun chunk. Vérifie DOC_OBJECTS et la PAR READ.")

    model = SentenceTransformer(EMB_MODEL_NAME)
    embs = model.encode([r["text"] for r in records],
                        batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embs = np.asarray(embs, dtype=np.float32)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    np.save(os.path.join(INDEX_DIR, "embeddings.npy"), embs)
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "records.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def load_index():
    """Charge l’index FAISS et les métadonnées."""
    embs = np.load(os.path.join(INDEX_DIR, "embeddings.npy"))
    index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
    recs = [json.loads(l) for l in open(os.path.join(INDEX_DIR, "records.jsonl"), "r")]
    model = SentenceTransformer(EMB_MODEL_NAME)
    return model, index, recs

def retrieve(question: str, top_k: int = TOP_K_DEFAULT) -> List[Dict[str, Any]]:
    """Recherche les passages les plus proches pour une question."""
    model, index, recs = load_index()
    q_emb = model.encode([question], normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb, top_k)
    hits = []
    for idx, score in zip(I[0], D[0]):
        hits.append({"doc": recs[idx]["doc"], "text": recs[idx]["text"], "score": float(score)})
    return hits

# ===================== PROMPT BUILDER =====================

def build_sources_block(hits: List[Dict[str, Any]]) -> str:
    """Construit le bloc de contexte pour le prompt."""
    lines = []
    for i, h in enumerate(hits, start=1):
        snippet = h["text"][:500].replace("\n", " ")
        lines.append(f"[S{i}] ({h['doc']}) {snippet}")
    return "\n".join(lines)

def build_runtime_prompt(question: str, hits: List[Dict[str, Any]]) -> str:
    """Assemble le prompt complet avec contexte et règles."""
    return RUNTIME_TEMPLATE.format(
        system=SYSTEM_RAG,
        question=question,
        sources_block=build_sources_block(hits)
    )

def rag_prepare(question: str, top_k: int = TOP_K_DEFAULT) -> Dict[str, Any]:
    """Prépare le prompt RAG et les métadonnées pour une question donnée."""
    hits = retrieve(question, top_k=top_k)
    return {
        "prompt": build_runtime_prompt(question, hits),
        "citations": [{"doc": h["doc"], "score": h["score"]} for h in hits],
        "sources_block": build_sources_block(hits),
        "system": SYSTEM_RAG,
        "question": question,
    }

# ===================== NETTOYAGE RÉPONSE LLM =====================

def clean_llm_answer(model_raw_text: str) -> Dict[str, Any]:
    """Parse et nettoie la réponse brute d’un LLM pour extraire summary, steps et citations."""
    raw = (model_raw_text or "").strip()
    summary, steps, cites = "", [], []

    # Extraire le Summary
    match_summary = re.search(r"Summary\s*[:\-]\s*(.+)", raw, re.IGNORECASE)
    if match_summary:
        summary = match_summary.group(1).strip()

    # Extraire les steps (Procedure)
    match_steps = re.search(r"Procedure\s*[:\-](.+?)(Citations|$)", raw, re.IGNORECASE | re.DOTALL)
    if match_steps:
        steps_block = match_steps.group(1).strip()
        steps = [line.strip(" -0123456789.").strip() for line in steps_block.splitlines() if line.strip()]

    # Extraire les Citations
    match_cites = re.search(r"Citations\s*[:\-]\s*(.+)", raw, re.IGNORECASE)
    if match_cites:
        cites = [c.strip() for c in re.split(r"[;,]", match_cites.group(1)) if c.strip()]

    return {
        "text": raw,
        "summary": summary,
        "steps": steps,
        "citations": cites,
    }


