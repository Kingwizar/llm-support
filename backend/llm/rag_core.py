# llm/rag_core.py
import os, json, re, requests, numpy as np, faiss
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from llm.prompt_builder import build_runtime_prompt
from llm.web_search import simple_web_search
from llm.ai_chat.voice_agent_prompt import build_voice_agent_prompt

import requests
# ==================== CONSTANTS ====================

INDEX_DIR = "rag_index"
os.makedirs(INDEX_DIR, exist_ok=True)

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K_DEFAULT = 4
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ==================== TEXT & CHUNKING ====================

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Découpe un texte long en morceaux avec chevauchement."""
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
    """Charge le FAISS index et ses métadonnées."""
    embs = np.load(os.path.join(INDEX_DIR, "embeddings.npy"))
    index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
    recs = [json.loads(l) for l in open(os.path.join(INDEX_DIR, "records.jsonl"), "r")]
    model = SentenceTransformer(EMB_MODEL_NAME)
    return model, index, recs

def retrieve(question: str, top_k: int = TOP_K_DEFAULT) -> List[Dict[str, Any]]:
    """Recherche les passages FAISS les plus proches."""
    model, index, recs = load_index()
    q_emb = model.encode([question], normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb, top_k)
    return [{"doc": recs[idx]["doc"], "text": recs[idx]["text"], "score": float(score)}
            for idx, score in zip(I[0], D[0])]

# ==================== BUILDER ====================

def rag_prepare(question: str, top_k: int = TOP_K_DEFAULT) -> Dict[str, Any]:
    """Prépare le prompt RAG complet pour la question donnée."""
    hits = retrieve(question, top_k=top_k)
    sources_block = "\n".join(f"[S{i}] ({h['doc']}) {h['text'][:400]}" for i, h in enumerate(hits, 1))
    return {
        "prompt": build_runtime_prompt(question, hits, sources_block),
        "citations": [{"doc": h["doc"], "score": h["score"]} for h in hits],
        "sources_block": sources_block,
        "question": question,
    }

# ==================== RAG + WEB ====================


def answer_with_rag_or_web(question: str, top_k: int = TOP_K_DEFAULT) -> Dict[str, Any]:
    """
    Version améliorée : effectue systématiquement une recherche web
    et combine les résultats web avec ceux du RAG (si pertinents).
    """

    # 🧩 Étape 1 — Recherche web systématique
    web_results = simple_web_search(question, num_results=3, full_content=True)
    web_block = "\n".join(
    f"[WEB] {r.get('title', '')}\nURL: {r.get('url', '')}\n{r.get('content', r.get('snippet', ''))[:2000]}"
    for r in web_results if not r.get("error")
    )

    web_citations = [{"doc": r.get("url", ""), "score": 0.3} for r in web_results if not r.get("error")]

    # 🧠 Étape 2 — Vérifie si l'index local est pertinent
    try:
        model, index, recs = load_index()
        q_emb = model.encode([question], normalize_embeddings=True).astype("float32")
        D, I = index.search(q_emb, top_k)
        max_sim = float(np.max(D))
    except Exception as e:
        print(f"[WARN] Impossible de charger l'index FAISS : {e}")
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

    # ⚙️ Étape 3 — Si l'index n’est pas pertinent, ne garder que le web
    if max_sim < SIM_THRESHOLD:
        print(f"[INFO] Similarité faible ({max_sim:.2f}) → réponse uniquement web")
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

    # 🧱 Étape 4 — Sinon, on combine RAG + Web
    pack = rag_prepare(question, top_k)
    hits = pack.get("citations", [])
    sources_block = pack.get("sources_block", "")

    # fusion propre
    combined_sources = sources_block + "\n\n# WEB SEARCH RESULTS\n" + web_block
    combined_citations = hits + web_citations

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





def query_ollama(prompt: str, model_name: str = "mistral-small:24b") -> str:
    url = "http://127.0.0.1:11434/api/chat"  # ✅ force IPv4
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "Tu es un assistant utile et précis."},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    headers = {"Content-Type": "application/json"}

    print("\n=== DEBUG OLLAMA REQUEST ===")
    print(f"→ URL     : {url}")
    print(f"→ Model   : {model_name}")
    print(f"→ Headers : {headers}")
    print("→ Payload :")
    print(json.dumps(payload, indent=2))
    print("============================\n")

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        print(f"[DEBUG] HTTP {response.status_code} from {url}")
        if not response.ok:
            print(f"[DEBUG] Response text: {response.text}")
        response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content", "").strip()
        print(f"[DEBUG] Response OK — length={len(content)} chars")
        return content
    except Exception as e:
        print(f"[ERROR contacting Ollama] {e}")
        return f"[Error contacting Ollama] {e}"





def test_rag_with_ollama(question: str):
    pack = answer_with_rag_or_web(question)
    print("=== QUESTION ===")
    print(question)
    print("\n=== PROMPT ===")
    print(pack["prompt"][:1000], "...")
    print("\n=== OLLAMA RESPONSE ===")
    answer = query_ollama(pack["prompt"], "mistral-small:24b")
    print(answer)

#--------------------------------Voice agent version ------------------------------------

# ===========================
# 🎤 AGENT VOCAL 3D UNREAL
# ===========================

def query_ollama_voice_agent(user_text: str, model_name: str = "mistral-small:24b") -> str:
    """
    Agent vocal pour personnage 3D Unreal.
    Utilise un prompt optimisé pour TTS + animation faciale.
    """
    url = "http://127.0.0.1:11434/api/chat"

    # 🔥 prompt optimisé pour la voix + animation MetaHuman
    prompt = build_voice_agent_prompt(user_text)

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": prompt}
        ],
        "stream": False
    }

    headers = {"Content-Type": "application/json"}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "").strip()

    except Exception as e:
        return f"[Erreur LLM Voice Agent] {e}"

