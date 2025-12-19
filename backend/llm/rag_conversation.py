import os, json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llm.rag_core import chunk_text, EMB_MODEL_NAME
from PyPDF2 import PdfReader
import docx
from io import BytesIO

BASE_DIR = "rag_index"

def conv_dir(conv_id: str) -> str:
    path = os.path.join(BASE_DIR, f"conv_{conv_id}")
    os.makedirs(path, exist_ok=True)
    return path

def load_conv_index(conv_id: str):
    path = conv_dir(conv_id)

    index_path = os.path.join(path, "faiss.index")
    recs_path = os.path.join(path, "records.jsonl")

    model = SentenceTransformer(EMB_MODEL_NAME)

    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        records = [json.loads(l) for l in open(recs_path)]
    else:
        index = faiss.IndexFlatIP(384)
        records = []

    return model, index, records


def extract_text(filename: str, data: bytes) -> str:
    if filename.endswith(".pdf"):
        reader = PdfReader(BytesIO(data))
        return "\n".join(p.extract_text() or "" for p in reader.pages)

    if filename.endswith(".docx"):
        d = docx.Document(BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs)

    return data.decode("utf-8", errors="ignore")

def index_file_for_conversation(conv_id: str, filename: str, text: str):
    model, index, records = load_conv_index(conv_id)

    chunks = chunk_text(text)
    if not chunks:
        return

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    ).astype("float32")

    index.add(embeddings)

    for c in chunks:
        records.append({
            "doc": filename or "Document utilisateur",
            "text": c
        })

    path = conv_dir(conv_id)
    faiss.write_index(index, os.path.join(path, "faiss.index"))

    with open(os.path.join(path, "records.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def retrieve_from_conversation(conv_id: str, question: str, top_k=4):
    model, index, records = load_conv_index(conv_id)

    if index.ntotal == 0:
        return []

    q_emb = model.encode([question], normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb, top_k)

    return [
        {
            "doc": records[i]["doc"],
            "text": records[i]["text"],
            "score": float(D[0][n])
        }
        for n, i in enumerate(I[0])
    ]
