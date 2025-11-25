import os
import json
import faiss
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

from voice_agent_prompt import build_voice_agent_prompt


# ==============================
# 🗂️ CHEMINS DES DOCUMENTS
# ==============================
RAG_FOLDER = "rag_docs/"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ==============================
# 🔤 CHARGEMENT EMBEDDING MODEL
# ==============================
embedder = SentenceTransformer(EMBED_MODEL_NAME)


# ==============================
# 📄 CHARGEMENT DES DOCUMENTS
# ==============================
def load_rag_documents(folder: str) -> List[Tuple[str, str]]:
    """
    Charge tous les fichiers du dossier RAG et retourne
    une liste (filename, text).
    """
    docs = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        if file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                docs.append((file, f.read()))

        elif file.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                docs.append((file, json.dumps(json.load(f), indent=2)))

    return docs


# Charger documents + chunking
def chunk_text(text: str, chunk_size=500, overlap=50):
    """
    Découpe en petits morceaux RAG-friendly.
    """
    text = text.replace("\n", " ").strip()
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap

    return chunks


def load_and_chunk_all_docs() -> Tuple[List[str], List[str]]:
    """
    Retourne :
    - la liste des chunks
    - la liste des métadonnées (nom de fichier)
    """
    raw_docs = load_rag_documents(RAG_FOLDER)
    chunks = []
    metadata = []

    for fname, text in raw_docs:
        parts = chunk_text(text)
        for p in parts:
            chunks.append(p)
            metadata.append(fname)

    return chunks, metadata


# ==============================
# 🧠 CONSTRUCTION INDEX FAISS
# ==============================
def build_faiss_index(chunks: List[str]):
    vectors = embedder.encode(chunks, convert_to_numpy=True)
    dim = vectors.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    return index


# ==============================
# 🔍 RECHERCHE RAG
# ==============================
def rag_search(query: str, index, chunks, metadata, top_k=4):
    q_vec = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_vec, top_k)

    results = []
    for idx in I[0]:
        results.append((chunks[idx], metadata[idx]))

    return results


# ==============================
# 🏗️ FUSION RAG + PROMPT AGENT
# ==============================
def build_rag_agent_prompt(user_message: str, index, chunks, metadata):
    """
    Combine :
    - RAG retrieved content
    - Ton prompt de personnage 3D
    - Le message utilisateur
    """
    rag_results = rag_search(user_message, index, chunks, metadata)

    retrieved_text = "\n\n".join(
        [f"[DOC: {meta}]\n{txt}" for txt, meta in rag_results]
    )

    final_message = (
        f"[RAG INFORMATION]\n{retrieved_text}\n\n"
        f"[USER QUESTION]\n{user_message}"
    )

    return build_voice_agent_prompt(final_message)


# ==============================
# ⚡ INITIALISATION (manuelle)
# ==============================
"""
Tu dois appeler manuellement ces lignes dans ton code :

chunks, metadata = load_and_chunk_all_docs()
index = build_faiss_index(chunks)

Puis :

prompt = build_rag_agent_prompt(user_input, index, chunks, metadata)
response = llm(prompt)
"""

