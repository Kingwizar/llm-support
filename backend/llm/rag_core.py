# ======================================================
# llm/rag_core.py
# Core RAG + LLM orchestration layer
# ======================================================

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


# ======================================================
# ----------------- CONSTANTS --------------------------
# ======================================================

# Base URL of the Ollama service (Docker service by default)
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://npone_ollama:11434")

# Ollama HTTP endpoints
OLLAMA_URL = f"{OLLAMA_BASE}/api/generate"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE}/api/chat"

# Directory storing FAISS index and metadata
INDEX_DIR = "rag_index"
os.makedirs(INDEX_DIR, exist_ok=True)

# Text chunking parameters for RAG indexing
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# Default number of retrieved chunks
TOP_K_DEFAULT = 4

# Sentence-transformer model used for embeddings
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ======================================================
# ----------------- TEXT CHUNKING ----------------------
# ======================================================

def chunk_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """
    Split a long text into overlapping chunks.

    - Normalizes whitespace
    - Uses a sliding window with overlap
    - Designed for semantic embedding (RAG)

    Returns a list of text chunks.
    """
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []

    chunks = []
    i = 0
    step = size - overlap

    while i < len(text):
        chunks.append(text[i:i + size])
        i += step

    return chunks


# ======================================================
# ----------------- FAISS CORE -------------------------
# ======================================================

def load_index():
    """
    Load the FAISS index and its associated metadata.

    Returns:
    - model: sentence-transformer embedding model
    - index: FAISS index for vector search
    - recs: metadata records aligned with embeddings
    """
    embs = np.load(os.path.join(INDEX_DIR, "embeddings.npy"))
    index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
    recs = [
        json.loads(l)
        for l in open(os.path.join(INDEX_DIR, "records.jsonl"), "r")
    ]

    model = SentenceTransformer(EMB_MODEL_NAME)
    return model, index, recs


def retrieve(
    question: str,
    top_k: int = TOP_K_DEFAULT
) -> List[Dict[str, Any]]:
    """
    Retrieve the most relevant text chunks for a question
    using vector similarity (cosine similarity).

    Returns a list of:
    - doc: document identifier
    - text: retrieved chunk
    - score: similarity score
    """
    model, index, recs = load_index()

    # Encode question into embedding space
    q_emb = model.encode(
        [question],
        normalize_embeddings=True
    ).astype("float32")

    # FAISS nearest neighbor search
    D, I = index.search(q_emb, top_k)

    return [
        {
            "doc": recs[idx]["doc"],
            "text": recs[idx]["text"],
            "score": float(score)
        }
        for idx, score in zip(I[0], D[0])
    ]


# ======================================================
# ----------------- RAG PREPARATION --------------------
# ======================================================

def rag_prepare(
    question: str,
    top_k: int = TOP_K_DEFAULT
) -> Dict[str, Any]:
    """
    Prepare a RAG package for the LLM.

    Steps:
    1. Retrieve top-k semantic chunks
    2. Build a formatted sources block
    3. Build the final LLM prompt

    Returns a structured dict used downstream.
    """
    hits = retrieve(question, top_k=top_k)

    # Build textual context injected into the prompt
    sources_block = "\n".join(
        f"[S{i}] ({h['doc']}) {h['text'][:400]}"
        for i, h in enumerate(hits, 1)
    )

    return {
        "prompt": build_runtime_prompt(
            question,
            hits,
            sources_block
        ),
        "citations": [
            {"doc": h["doc"], "score": h["score"]}
            for h in hits
        ],
        "sources_block": sources_block,
        "question": question,
    }


# ======================================================
# ----------------- RAG + WEB STRATEGY -----------------
# ======================================================

def answer_with_rag_or_web(
    question: str,
    top_k: int = TOP_K_DEFAULT
) -> Dict[str, Any]:
    """
    Main decision-making function for knowledge retrieval.

    Strategy:
    1. Always perform a web search (cheap, fallback-safe)
    2. Try FAISS similarity to detect internal knowledge relevance
    3. If similarity is low → web only
    4. If similarity is high → RAG + web augmentation

    This function DOES NOT call the LLM directly.
    It only prepares the final prompt and sources.
    """

    # ---------- WEB SEARCH ----------
    web_results = simple_web_search(
        question,
        num_results=3,
        full_content=True
    )

    web_block = "\n".join(
        f"[WEB] {r.get('title','')}\n"
        f"URL: {r.get('url','')}\n"
        f"{r.get('content', r.get('snippet',''))[:2000]}"
        for r in web_results
        if not r.get("error")
    )

    web_citations = [
        {"doc": r.get("url", ""), "score": 0.3}
        for r in web_results
        if not r.get("error")
    ]

    # ---------- FAISS SIMILARITY CHECK ----------
    try:
        model, index, _ = load_index()
        q_emb = model.encode(
            [question],
            normalize_embeddings=True
        ).astype("float32")

        D, _ = index.search(q_emb, top_k)
        max_sim = float(np.max(D))

    except Exception as e:
        # FAISS unavailable → web-only fallback
        return {
            "prompt": build_runtime_prompt(
                question,
                web_citations,
                web_block
            ),
            "citations": web_citations,
            "sources_block": web_block,
            "question": question,
            "from_rag": False,
            "from_web": True,
            "similarity": None,
        }

    # ---------- DECISION THRESHOLD ----------
    SIM_THRESHOLD = 0.35

    if max_sim < SIM_THRESHOLD:
        # Internal knowledge not relevant enough
        prompt = build_runtime_prompt(
            question,
            web_citations,
            web_block
        )

        return {
            "prompt": prompt,
            "citations": web_citations,
            "sources_block": web_block,
            "question": question,
            "from_rag": False,
            "from_web": True,
            "similarity": max_sim,
        }

    # ---------- RAG + WEB MERGE ----------
    pack = rag_prepare(question, top_k)

    combined_sources = (
        pack["sources_block"]
        + "\n\n# WEB SEARCH RESULTS\n"
        + web_block
    )

    combined_citations = (
        pack["citations"]
        + web_citations
    )

    prompt = build_runtime_prompt(
        question,
        combined_citations,
        combined_sources
    )

    return {
        "prompt": prompt,
        "citations": combined_citations,
        "sources_block": combined_sources,
        "question": question,
        "from_rag": True,
        "from_web": True,
        "similarity": max_sim,
    }


# ======================================================
# ----------------- OLLAMA (TEXT) ----------------------
# ======================================================

def query_ollama(
    prompt: str,
    model_name: str = "qwen14b_llm"
) -> str:
    """
    Send a fully-built prompt to Ollama (no streaming).

    This is the ONLY place where:
    - The LLM is actually called
    - The prompt leaves the backend

    All RAG logic must be completed BEFORE this call.
    """
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    # Debug visibility for prompt inspection
    print("\n================ PROMPT SENT TO OLLAMA ================")
    print(f"MODEL      : {model_name}")
    print(f"PROMPT LEN : {len(prompt)} chars")
    print("------------------------------------------------------")
    print(prompt)
    print("======================================================\n")

    try:
        r = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"[Ollama HTTP Error] {e}"


# ======================================================
# ----------------- CLI / LOCAL TEST -------------------
# ======================================================

def test_rag_with_ollama(question: str):
    """
    Local debugging helper:
    - Builds RAG/web prompt
    - Sends it to Ollama
    - Prints the response
    """
    pack = answer_with_rag_or_web(question)

    print("\n=== QUESTION ===")
    print(question)

    answer = query_ollama(pack["prompt"], "qwen14b_llm")

    print("\n=== OLLAMA RESPONSE ===")
    print(answer)


def query_ollama_local(prompt: str, model="qwen14b_llm"):
    """
    Alternative Ollama invocation via CLI (subprocess).
    Useful for debugging without HTTP.
    """
    process = subprocess.Popen(
        ["ollama", "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    out, _ = process.communicate(prompt)
    return out.strip()


# ======================================================
# ----------------- VOICE AGENT ------------------------
# ======================================================

def query_ollama_voice_agent(
    user_text: str,
    model_name: str = "qwen14b_llm"
) -> str:
    """
    Specialized LLM call for voice/3D agents.

    Differences from standard chat:
    - Uses /api/chat
    - Injects a system-level voice-agent prompt
    - Optimized for short, spoken responses
    """
    prompt = build_voice_agent_prompt(user_text)

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": prompt}
        ],
        "stream": False
    }

    try:
        r = requests.post(
            OLLAMA_CHAT_URL,
            json=payload,
            timeout=60
        )
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "").strip()
    except Exception as e:
        return f"[LLM Voice Agent Error] {e}"
