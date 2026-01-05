# llm/rag_core.py
import os
import json
import re
import requests
import numpy as np
import faiss
import subprocess
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from io import BytesIO
from llm.prompt_builder import build_runtime_prompt
from llm.web_search import simple_web_search
from llm.ai_chat.voice_agent_prompt import build_voice_agent_prompt

# ==================== CONSTANTS ====================
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

OLLAMA_URL = f"{OLLAMA_BASE}/api/generate"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE}/api/chat"

INDEX_DIR = "rag_index"
os.makedirs(INDEX_DIR, exist_ok=True)

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K_DEFAULT = 4
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OLLAMA_URL = "http://ollama:11434/api/generate"
# ==================== TEXT & CHUNKING ====================

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    chunks, i = [], 0
    step = size - overlap
    while i < len(text):
        chunks.append(text[i:i + size])
        i += step
    return chunks

# ==================== FAISS RAG CORE ====================

def load_index():
    embs = np.load(os.path.join(INDEX_DIR, "embeddings.npy"))
    index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
    recs = [json.loads(l) for l in open(os.path.join(INDEX_DIR, "records.jsonl"), "r")]
    model = SentenceTransformer(EMB_MODEL_NAME)
    return model, index, recs

def retrieve(question: str, top_k: int = TOP_K_DEFAULT) -> List[Dict[str, Any]]:
    model, index, recs = load_index()
    q_emb = model.encode([question], normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb, top_k)
    return [
        {"doc": recs[idx]["doc"], "text": recs[idx]["text"], "score": float(score)}
        for idx, score in zip(I[0], D[0])
    ]

# ==================== RAG PREP ====================

def rag_prepare(question: str, top_k: int = TOP_K_DEFAULT) -> Dict[str, Any]:
    hits = retrieve(question, top_k=top_k)
    sources_block = "\n".join(
        f"[S{i}] ({h['doc']}) {h['text'][:400]}"
        for i, h in enumerate(hits, 1)
    )
    return {
        "prompt": build_runtime_prompt(question, hits, sources_block),
        "citations": [{"doc": h["doc"], "score": h["score"]} for h in hits],
        "sources_block": sources_block,
        "question": question,
    }

# ==================== RAG + WEB ====================

def answer_with_rag_or_web(question: str, top_k: int = TOP_K_DEFAULT) -> Dict[str, Any]:
    web_results = simple_web_search(question, num_results=3, full_content=True)

    web_block = "\n".join(
        f"[WEB] {r.get('title','')}\nURL: {r.get('url','')}\n{r.get('content', r.get('snippet',''))[:2000]}"
        for r in web_results if not r.get("error")
    )

    web_citations = [
        {"doc": r.get("url",""), "score": 0.3}
        for r in web_results if not r.get("error")
    ]

    try:
        model, index, _ = load_index()
        q_emb = model.encode([question], normalize_embeddings=True).astype("float32")
        D, _ = index.search(q_emb, top_k)
        max_sim = float(np.max(D))
    except Exception as e:
        print(f"[WARN] FAISS indisponible : {e}")
        return {
            "prompt": build_runtime_prompt(question, web_citations, web_block),
            "citations": web_citations,
            "sources_block": web_block,
            "question": question,
            "from_rag": False,
            "from_web": True,
            "similarity": None,
        }

    SIM_THRESHOLD = 0.35

    if max_sim < SIM_THRESHOLD:
        print(f"[INFO] Similarité FAISS faible ({max_sim:.2f}) → Web only")
        prompt = build_runtime_prompt(question, web_citations, web_block)
        return {
            "prompt": prompt,
            "citations": web_citations,
            "sources_block": web_block,
            "question": question,
            "from_rag": False,
            "from_web": True,
            "similarity": max_sim,
        }

    pack = rag_prepare(question, top_k)
    combined_sources = pack["sources_block"] + "\n\n# WEB SEARCH RESULTS\n" + web_block
    combined_citations = pack["citations"] + web_citations

    prompt = build_runtime_prompt(question, combined_citations, combined_sources)
    return {
        "prompt": prompt,
        "citations": combined_citations,
        "sources_block": combined_sources,
        "question": question,
        "from_rag": True,
        "from_web": True,
        "similarity": max_sim,
    }

# ============================
# OLLAMA HTTP (NO STREAM)
# ============================

def query_ollama(prompt: str, model_name: str = "qwen14b_llm") -> str:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    # ===== DEBUG PROMPT =====
    print("\n================ PROMPT SENT TO OLLAMA ================")
    print(f"MODEL      : {model_name}")
    print(f"PROMPT LEN : {len(prompt)} chars")
    print("------------------------------------------------------")
    print(prompt)
    print("======================================================\n")

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=300)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"[Erreur Ollama HTTP] {e}"

# ==================== TEST ====================

def test_rag_with_ollama(question: str):
    pack = answer_with_rag_or_web(question)

    print("\n=== QUESTION ===")
    print(question)

    answer = query_ollama(pack["prompt"], "qwen14b_llm")

    print("\n=== OLLAMA RESPONSE ===")
    print(answer)

# ==================== LOCAL CLI (OPTIONNEL) ====================

def query_ollama_local(prompt: str, model="qwen14b_llm"):
    process = subprocess.Popen(
        ["ollama", "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    out, _ = process.communicate(prompt)
    return out.strip()

# ===========================
# AGENT VOCAL 3D UNREAL
# ===========================

def query_ollama_voice_agent(user_text: str, model_name: str = "qwen14b_llm") -> str:
    url = OLLAMA_CHAT_URL

    prompt = build_voice_agent_prompt(user_text)

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": prompt}
        ],
        "stream": False
    }

    try:
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "").strip()
    except Exception as e:
        return f"[Erreur LLM Voice Agent] {e}"
