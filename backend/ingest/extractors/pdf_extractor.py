import fitz


# PDF Extractor (extractors/pdf_extractor.py)
def extract_pdf_text(path):
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return {"type": "pdf", "text": text.strip()}