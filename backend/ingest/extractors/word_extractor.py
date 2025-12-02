from PIL import Image
from docx import Document
import pytesseract
# WORD Extractor (extractors/word_extractor.py)
def extract_word_text(path):
    doc = Document(path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return {"type": "docx", "text": text.strip()}