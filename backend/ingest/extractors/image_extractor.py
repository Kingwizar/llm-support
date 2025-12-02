from PIL import Image
import pytesseract
import os

def extract_image_text(path):
    """
    Extrait le texte depuis une image (OCR Tesseract) et génère un .txt
    dans le même dossier temporaire où se trouve l'image.
    """

    # 1️⃣ Lecture image
    image = Image.open(path)

    # 2️⃣ Extraction OCR
    text = pytesseract.image_to_string(image, lang="eng+fra")
    text = text.strip()

    # 3️⃣ Construction du chemin du .txt dans le même dossier que l'image
    base_path, ext = os.path.splitext(path)
    txt_path = base_path + ".txt"  # ex: /tmp/image.png → /tmp/image.txt

    try:
        # 4️⃣ Sauvegarde du texte dans le fichier TXT
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"📄 Fichier texte image généré : {txt_path}")
    except Exception as e:
        print(f"⚠️ Impossible d'écrire dans {txt_path} : {e}")

    # 5️⃣ Retour pour le pipeline RAG
    return {
        "type": "image",
        "text": text
    }
