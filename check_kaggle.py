# check_kaggle.py
import os, sys, json, textwrap
from typing import List, Optional
import pandas as pd

# Heuristiques simples pour repérer les colonnes utiles
Q_KEYS = ["question","query","issue","problem","description","subject","ticket","body","text","message"]
A_KEYS = ["answer","resolution","response","reply","solution","agent","handler"]
I_KEYS = ["intent","category","type","class","label","labels","queue","topic"]

def pick_col(cols: List[str], keys: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for k in low:
        if any(key in k for key in keys):
            return low[k]
    return None

def preview_df(path: str, n=2):
    # lecture par extension
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".tsv"]:
        sep = "," if ext == ".csv" else "\t"
        df = pd.read_csv(path, nrows=200)  # petit échantillon
    elif ext in [".jsonl", ".json"]:
        try:
            # jsonl
            with open(path, "r", encoding="utf-8") as f:
                rows = [json.loads(next(f)) for _ in range(200)]
            df = pd.DataFrame(rows)
        except Exception:
            # json normal
            obj = json.load(open(path, "r", encoding="utf-8"))
            if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
                df = pd.DataFrame(obj["data"])
            else:
                df = pd.DataFrame(obj if isinstance(obj, list) else [obj])
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Extension non supportée: {ext}")
    return df

def tronq(x, n=120):
    if pd.isna(x):
        return None
    s = str(x).replace("\n", " ").replace("\r", " ")
    return (s[: n - 3] + "...") if len(s) > n else s

def analyze(path: str, dataset_id: str):
    print("=" * 120)
    print(f"FILE: {path}")
    try:
        df = preview_df(path)
    except Exception as e:
        print(f"⚠️ Impossible de lire {path}: {e}")
        return

    cols = list(df.columns)
    print(f"Colonnes ({len(cols)}): {cols}")

    # Heuristiques
    q_col = pick_col(cols, Q_KEYS)
    a_col = pick_col(cols, A_KEYS)
    i_col = pick_col(cols, I_KEYS)

    print(f"→ Détection heuristique:")
    print(f"   user_text: {q_col}")
    print(f"   agent_text: {a_col}")
    print(f"   intent/category: {i_col}")

    # Exemples
    print("-" * 120)
    print("Aperçu (2 lignes):")
    for _, row in df.head(2).iterrows():
        ex = {c: tronq(row[c]) for c in cols}
        print(ex)

    # Squelette de mapper proposé
    mapper_stub = f"""
# Squelette de mapper dédié pour {dataset_id}
def map_kaggle_custom_row(row, split):
    return mk_conv_item(
        "kaggle",
        "{dataset_id}",
        split,
        row.get("id") or row.get("ID") or row.get("Ticket_ID") or row.name,  # ajuste l'ID si tu as une vraie colonne
        row.get("{q_col}") if "{q_col}" != "None" else None,
        row.get("{a_col}") if "{a_col}" != "None" else None,
        intent=row.get("{i_col}") if "{i_col}" != "None" else None,
        category=row.get("{i_col}") if "{i_col}" != "None" else None,  # ou remplace par une autre colonne
        tags=["kaggle","it","support"]
    )
"""
    print("-" * 120)
    print("Mapper dédié suggéré :")
    print(textwrap.dedent(mapper_stub))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python check_kaggle.py <dataset_id> <path1> [<path2> ...]")
        print("Ex:")
        print(r'  python check_kaggle.py "parthpatil256/it-support-ticket-data" .\data\it_support.csv')
        sys.exit(1)

    dataset_id = sys.argv[1]
    for p in sys.argv[2:]:
        analyze(p, dataset_id)
