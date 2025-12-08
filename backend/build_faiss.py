import os, json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llm.rag_core import chunk_text

SOURCE_DIR = "rag_docs"
INDEX_DIR = "rag_index"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

records = []
embeddings = []

for fname in os.listdir(SOURCE_DIR):
    path = os.path.join(SOURCE_DIR, fname)
    if not os.path.isfile(path):
        continue

    with open(path, "r", encoding="utf8") as f:
        text = f.read().strip()

    chunks = chunk_text(text)
    for c in chunks:
        embeddings.append(model.encode(c, normalize_embeddings=True))
        records.append({"doc": fname, "text": c})

# convert embeddings
embeddings = np.array(embeddings, dtype="float32")

# build FAISS
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# save FAISS + embeddings
faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
np.save(os.path.join(INDEX_DIR, "embeddings.npy"), embeddings)

# save JSONL
with open(os.path.join(INDEX_DIR, "records.jsonl"), "w", encoding="utf8") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

print("✅ RAG reconstruit. Documents indexés :", len(records))
