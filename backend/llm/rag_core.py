# llm/rag_core.py
import os, json, re, requests, numpy as np, faiss
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from llm.prompt_builder import build_runtime_prompt
from llm.web_search import simple_web_search

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

def answer_with_rag_or_web(question: str) -> Dict[str, Any]:
    """Essaie RAG d’abord, sinon web search."""
    pack = rag_prepare(question)
    if not pack["citations"]:
        results = simple_web_search(question)
        pack["sources_block"] = "\n".join(f"[WEB] {r['title']} — {r['snippet']}" for r in results)
        pack["citations"] = [{"doc": r["url"], "score": 0.4} for r in results]
    return pack
