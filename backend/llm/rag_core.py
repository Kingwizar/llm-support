# llm/rag_core.py
import os, json, re, requests, numpy as np, faiss
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from llm.prompt_builder import build_runtime_prompt
from llm.web_search import simple_web_search
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

from llm.web_search import simple_web_search
from llm.prompt_builder import build_runtime_prompt

def answer_with_rag_or_web(question: str, top_k: int = TOP_K_DEFAULT) -> Dict[str, Any]:
    """
    Si la question est peu liée au RAG (sim < seuil), on ignore l'index et on lance une recherche web.
    Sinon, on fait RAG + web si besoin.
    """
    model, index, recs = load_index()
    q_emb = model.encode([question], normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb, top_k)
    max_sim = float(np.max(D))

    SIM_THRESHOLD = 0.35  # Ajustable : plus haut = plus strict

    # 🟡 Cas 1 : Question hors domaine → uniquement web search
    if max_sim < SIM_THRESHOLD:
        print(f"[INFO] Similarité faible ({max_sim:.2f}) → RAG ignoré, utilisation du Web")
        web_results = simple_web_search(question)
        web_block = "\n".join(
            f"[WEB] {r.get('title', '')} — {r.get('snippet', '')}" for r in web_results if not r.get("error")
        )
        citations = [{"doc": r.get("url", ""), "score": 0.3} for r in web_results if not r.get("error")]
        prompt = build_runtime_prompt(question, citations, web_block)
        return {
            "prompt": prompt,
            "citations": citations,
            "sources_block": web_block,
            "question": question,
            "from_rag": False,
            "from_web": True,
            "similarity": max_sim,
        }

    # 🟢 Cas 2 : Pertinent → RAG + éventuellement Web
    pack = rag_prepare(question, top_k)
    hits = pack.get("citations", [])
    sources_block = pack.get("sources_block", "")

    # Compléter si peu de résultats
    if len(hits) < 2:
        web_results = simple_web_search(question)
        web_block = "\n".join(
            f"[WEB] {r.get('title', '')} — {r.get('snippet', '')}" for r in web_results if not r.get("error")
        )
        sources_block += "\n" + web_block
        hits.extend(
            [{"doc": r.get("url", ""), "score": 0.3} for r in web_results if not r.get("error")]
        )

    prompt = build_runtime_prompt(question, hits, sources_block)
    return {
        "prompt": prompt,
        "citations": hits,
        "sources_block": sources_block,
        "question": question,
        "from_rag": True,
        "from_web": len(hits) < 2,
        "similarity": max_sim,
    }




def query_ollama(prompt: str, model_name: str = "mistral") -> str:
    """
    Envoie un prompt à Ollama (modèle local comme mistral) et renvoie la réponse textuelle.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        

        return data.get("response", "").strip()
        
    except Exception as e:
        return f"[Error contacting Ollama] {e}"

def test_rag_with_ollama(question: str):
    pack = answer_with_rag_or_web(question)
    print("=== QUESTION ===")
    print(question)
    print("\n=== PROMPT ===")
    print(pack["prompt"][:1000], "...")
    print("\n=== OLLAMA RESPONSE ===")
    answer = query_ollama(pack["prompt"], "mistral")
    print(answer)

