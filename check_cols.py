# check_cols.py
import textwrap
from typing import List, Optional

def t(s: Optional[str], n=120):
    """Tronque proprement pour l'affichage."""
    if s is None:
        return None
    s = str(s).replace("\n", " ").replace("\r", " ")
    return (s[: n - 3] + "...") if len(s) > n else s

def show_hf(name: str, splits: List[str] = None, samples: int = 2):
    from datasets import get_dataset_split_names, load_dataset

    print("=" * 120)
    print(f"DATASET: {name}")
    try:
        avail = get_dataset_split_names(name)
        print(f"Splits disponibles: {avail}")
    except Exception as e:
        print(f"Impossible de récupérer les splits: {e}")
        avail = []

    use_splits = splits or (avail if avail else ["train"])
    for sp in use_splits:
        print("-" * 120)
        print(f"SPLIT: {sp}")
        try:
            ds = load_dataset(name, split=sp)
        except Exception as e:
            print(f"  ⚠️  load_dataset échoue: {e}")
            continue

        print(f"  Lignes: {len(ds)}")
        print(f"  Colonnes: {ds.column_names}")

        for i in range(min(samples, len(ds))):
            row = ds[i]
            # affiche un aperçu compact
            preview = {k: t(row.get(k)) for k in ds.column_names}
            print(f"  Exemple {i}: {preview}")

if __name__ == "__main__":
    # Liste tes datasets ici
    HF_DATASETS = [
        "Talhat/Customer_IT_Support",
        "VivKatara/customer-support-it-dataset-split",
        "harishkotra/technical-support-dataset",
        "MakTek/Customer_support_faqs_dataset",
        "balawin/FAQ_Support",
    ]

    # Splits à tester (None = tous les splits annoncés par le hub)
    SPLITS_OVERRIDE = {
        # "Talhat/Customer_IT_Support": ["train", "test"],
        # "VivKatara/customer-support-it-dataset-split": ["train", "validation", "test"],
        # "harishkotra/technical-support-dataset": ["train"],
        # "MakTek/Customer_support_faqs_dataset": ["train"],
        # "balawin/FAQ_Support": ["train"],
    }

    for ds_name in HF_DATASETS:
        show_hf(ds_name, splits=SPLITS_OVERRIDE.get(ds_name))
