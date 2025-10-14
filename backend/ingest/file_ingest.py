# file_ingestion.py
import os
from typing import Dict, Any


def extract_text_from_file(file_path: str) -> Dict[str, Any]:
    """Détecte le type de fichier et en extrait le texte brut."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        from extractors.pdf_extractor import extract_pdf_text
        return extract_pdf_text(file_path)

    elif ext in [".doc", ".docx"]:
        from extractors.word_extractor import extract_word_text
        return extract_word_text(file_path)

    elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        from extractors.image_extractor import extract_image_text
        return extract_image_text(file_path)

    else:
        raise ValueError(f"Format de fichier non pris en charge : {ext}")


