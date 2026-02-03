# ======================================================
# Conversation-level RAG indexing and retrieval
# ======================================================

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llm.rag_core import chunk_text, EMB_MODEL_NAME
from PyPDF2 import PdfReader
import docx
from io import BytesIO


# ======================================================
# ----------------- BASE DIRECTORY ---------------------
# ======================================================

# Root directory where all conversation-specific RAG indexes are stored
BASE_DIR = "rag_index"


# ======================================================
# ----------------- PATH HELPERS -----------------------
# ======================================================

def conv_dir(conv_id: str) -> str:
    """
    Build and ensure the directory dedicated to a conversation.

    Each conversation has its own isolated FAISS index:
    rag_index/conv_<conversation_id>/
    """
    path = os.path.join(BASE_DIR, f"conv_{conv_id}")
    os.makedirs(path, exist_ok=True)
    return path


# ======================================================
# ----------------- INDEX LOADING ----------------------
# ======================================================

def load_conv_index(conv_id: str):
    """
    Load or initialize the FAISS index for a given conversation.

    Returns:
    - model   : sentence-transformer used for embeddings
    - index   : FAISS index (inner product / cosine similarity)
    - records : list of text chunks metadata aligned with embeddings
    """
    path = conv_dir(conv_id)

    index_path = os.path.join(path, "faiss.index")
    recs_path = os.path.join(path, "records.jsonl")

    # Embedding model (shared with global RAG)
    model = SentenceTransformer(EMB_MODEL_NAME)

    if os.path.exists(index_path):
        # Existing conversation index
        index = faiss.read_index(index_path)
        records = [json.loads(l) for l in open(recs_path)]
    else:
        # New empty index (384 = MiniLM embedding size)
        index = faiss.IndexFlatIP(384)
        records = []

    return model, index, records


# ======================================================
# ----------------- TEXT EXTRACTION --------------------
# ======================================================

def extract_text(filename: str, data: bytes) -> str:
    """
    Extract raw text from uploaded files.

    Supported formats:
    - PDF
    - DOCX
    - Plain text (fallback)
    """
    if filename.endswith(".pdf"):
        reader = PdfReader(BytesIO(data))
        return "\n".join(
            p.extract_text() or "" for p in reader.pages
        )

    if filename.endswith(".docx"):
        d = docx.Document(BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs)

    # Fallback for text-like files
    return data.decode("utf-8", errors="ignore")


# ======================================================
# ----------------- INDEXING PIPELINE ------------------
# ======================================================

def index_file_for_conversation(
    conv_id: str,
    filename: str,
    text: str
):
    """
    Index a document into the conversation-level FAISS index.

    Steps:
    1. Load or create the conversation index
    2. Split text into overlapping chunks
    3. Generate embeddings for each chunk
    4. Add embeddings to FAISS
    5. Persist index and metadata on disk
    """
    model, index, records = load_conv_index(conv_id)

    # Split document into semantic chunks
    chunks = chunk_text(text)
    if not chunks:
        return

    # Compute embeddings (cosine similarity via normalization)
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    ).astype("float32")

    # Add embeddings to FAISS index
    index.add(embeddings)

    # Store aligned metadata for each chunk
    for c in chunks:
        records.append({
            "doc": filename or "User document",
            "text": c
        })

    # Persist index and metadata
    path = conv_dir(conv_id)
    faiss.write_index(index, os.path.join(path, "faiss.index"))

    with open(os.path.join(path, "records.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ======================================================
# ----------------- RETRIEVAL --------------------------
# ======================================================

def retrieve_from_conversation(
    conv_id: str,
    question: str,
    top_k: int = 4
):
    """
    Retrieve the most relevant chunks from a conversation index.

    Used to prioritize user-uploaded documents over
    global RAG or web sources.
    """
    model, index, records = load_conv_index(conv_id)

    # No documents indexed yet
    if index.ntotal == 0:
        return []

    # Encode the question into embedding space
    q_emb = model.encode(
        [question],
        normalize_embeddings=True
    ).astype("float32")

    # Nearest neighbor search
    D, I = index.search(q_emb, top_k)

    return [
        {
            "doc": records[i]["doc"],
            "text": records[i]["text"],
            "score": float(D[0][n])
        }
        for n, i in enumerate(I[0])
    ]
