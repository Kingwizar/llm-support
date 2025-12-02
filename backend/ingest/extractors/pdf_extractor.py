import fitz
import os

# PDF Extractor (extractors/pdf_extractor.py)
def extract_pdf_text(path):
    """
    Extrait le texte d'un PDF et génère un fichier .txt avec le même nom.
    Retourne également un dictionnaire {type, text} compatible avec ton pipeline RAG.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ PDF introuvable : {path}")

    text = ""

    # 1️⃣ Lecture PDF
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"

    text = text.strip()

    # 2️⃣ Création du fichier .txt
    txt_path = path.rsplit(".", 1)[0] + ".txt"  # même nom que le PDF
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"📄 Fichier texte généré : {txt_path}")
    except Exception as e:
        print(f"⚠️ Impossible d’écrire le fichier TXT : {e}")

    # 3️⃣ Retour compatible avec ta logique RAG
    return {
        "type": "pdf",
        "text": text
    }
