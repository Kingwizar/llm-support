# ds_pipeline/ingest.py
import os
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, TEXT
from pymongo.errors import BulkWriteError
from datasets import load_dataset
from tqdm import tqdm

from .mappers import (
    mk_conv_item,
    map_hf_talhat,
    map_hf_vivkatara,
    map_hf_harishkotra,
    map_hf_maktek,
    map_hf_balawin,
    map_kaggle_parthpatil,
    map_kaggle_tobiasbueck,
    map_kaggle_adisongoh,
)

load_dotenv()
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME   = os.getenv("DB_NAME", "llm_support")
COLL_NAME = os.getenv("COLLECTION", "conversations")

mongo = MongoClient(MONGO_URI)
col = mongo[DB_NAME][COLL_NAME]

# -------------------- util --------------------
def insert_batch(batch, inserted_counter):
    if not batch:
        return inserted_counter
    try:
        res = col.insert_many(batch, ordered=False)
        inserted_counter += len(res.inserted_ids)
    except BulkWriteError as e:
        inserted_counter += e.details.get("nInserted", 0)
    finally:
        batch.clear()
    return inserted_counter

def ensure_indexes():
    col.create_index([("conversation_id", ASCENDING)], unique=True, name="uniq_conv_id")
    col.create_index([("source.dataset", ASCENDING), ("source.split", ASCENDING)], name="src_split")
    col.create_index([("labels.intent", ASCENDING)], name="intent_idx")
    col.create_index([("messages.text", TEXT)], name="msg_text_idx")

# -------------------- HF dispatch --------------------
HF_MAP = {
    "Talhat/Customer_IT_Support": map_hf_talhat,
    "VivKatara/customer-support-it-dataset-split": map_hf_vivkatara,
    "harishkotra/technical-support-dataset": map_hf_harishkotra,
    "MakTek/Customer_support_faqs_dataset": map_hf_maktek,
    "balawin/FAQ_Support": map_hf_balawin,
}

def ingest_hf(ds_name: str, splits):
    if ds_name not in HF_MAP:
        raise ValueError(f"Aucun mapper défini pour {ds_name}. Datasets disponibles: {list(HF_MAP.keys())}")
    mapper = HF_MAP[ds_name]
    total_inserted = 0
    for split in splits:
        ds = load_dataset(ds_name, split=split)
        batch, inserted = [], 0
        for row in tqdm(ds, desc=f"{ds_name}:{split}"):
            doc = mapper(row, split)
            if doc and doc.get("messages"):
                batch.append(doc)
            if len(batch) >= 1000:
                inserted = insert_batch(batch, inserted)
        inserted = insert_batch(batch, inserted)
        total_inserted += inserted
        print(f"[{ds_name}:{split}] insérés: {inserted}")
    print(f"[{ds_name}] total insérés: {total_inserted}")

# -------------------- Kaggle CSV générique --------------------
def ingest_kaggle_csv(csv_path: str, dataset_id: str, split="train"):
    df = pd.read_csv(csv_path)
    lower = {c.lower(): c for c in df.columns}
    def pick(candidates):
        for lc, orig in lower.items():
            if any(key in lc for key in candidates):
                return orig
        return None
    q_col = pick(["question", "query", "issue", "problem", "description", "subject", "text", "body", "message"])
    a_col = pick(["answer", "resolution", "response", "reply", "solution"])
    i_col = pick(["intent", "category", "type", "class", "label", "labels", "queue", "topic"])

    batch, inserted = [], 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{dataset_id}:{split}"):
        row.name = idx
        user_text  = str(row[q_col]) if q_col and pd.notna(row[q_col]) else None
        agent_text = str(row[a_col]) if a_col and pd.notna(row[a_col]) else None
        intent     = str(row[i_col]) if i_col and pd.notna(row[i_col]) else None
        doc = mk_conv_item("kaggle", dataset_id, split, idx, user_text, agent_text,
                           intent=intent, category=intent, tags=["kaggle"])
        if doc and doc.get("messages"):
            batch.append(doc)
        if len(batch) >= 1000:
            inserted = insert_batch(batch, inserted)
    inserted = insert_batch(batch, inserted)
    print(f"[{dataset_id}:{split}] insérés: {inserted}")

# -------------------- Kaggle CSV spécialisés --------------------
def ingest_kaggle_custom(csv_path: str, dataset_id: str, mapper, split="train"):
    df = pd.read_csv(csv_path)
    # Nettoyage de colonnes d'index parasites
    for noisy in ("Unnamed: 0", "index", "Index"):
        if noisy in df.columns:
            df = df.drop(columns=[noisy])
    batch, inserted = [], 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{dataset_id}:{split}"):
        row.name = idx
        doc = mapper(row, split)
        if doc and doc.get("messages"):
            batch.append(doc)
        if len(batch) >= 1000:
            inserted = insert_batch(batch, inserted)
    inserted = insert_batch(batch, inserted)
    print(f"[{dataset_id}:{split}] insérés: {inserted}")

# -------------------- main CLI --------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensure-indexes", action="store_true")
    parser.add_argument("--hf", nargs="*", help="Liste de datasets HF à ingérer (owner/name)")
    parser.add_argument("--hf-splits", nargs="*", default=["train"])
    parser.add_argument("--kaggle", nargs="*", help="Paires owner/name=./file.csv (ingestion générique)")
    parser.add_argument("--kaggle-special", nargs="*", help="Paires owner/name=./file.csv (mappers dédiés)")
    args = parser.parse_args()

    if args.ensure_indexes:
        ensure_indexes()
        print("Indexes OK")

    if args.hf:
        for ds_name in args.hf:
            ingest_hf(ds_name, args.hf_splits)

    if args.kaggle:
        for pair in args.kaggle:
            if "=" not in pair:
                print(f"Format invalide pour --kaggle: {pair} (attendu owner/name=csv_path)")
                continue
            ds_id, csv_path = pair.split("=", 1)
            ingest_kaggle_csv(csv_path, ds_id, split="train")

    if args.kaggle_special:
        for pair in args.kaggle_special:
            if "=" not in pair:
                print(f"Format invalide pour --kaggle-special: {pair} (attendu owner/name=csv_path)")
                continue
        for pair in args.kaggle_special:
            ds_id, csv_path = pair.split("=", 1)
            if ds_id == "parthpatil256/it-support-ticket-data":
                ingest_kaggle_custom(csv_path, ds_id, map_kaggle_parthpatil, split="train")
            elif ds_id in (
                "tobiasbueck/multilingual-customer-support-tickets",
                "tobiasbueck/multilingual-customer-support-tickets/data",
            ):
                ingest_kaggle_custom(csv_path, ds_id, map_kaggle_tobiasbueck, split="train")
            elif ds_id == "adisongoh/it-service-ticket-classification-dataset":
                ingest_kaggle_custom(csv_path, ds_id, map_kaggle_adisongoh, split="train")
            else:
                print(f"Aucun mapper spécial défini pour {ds_id}")

    print("Import terminé.")
