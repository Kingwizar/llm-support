from PIL import Image
import pytesseract

def extract_image_text(path):
    image = Image.open(path)
    text = pytesseract.image_to_string(image, lang="eng+fra")
    return {"type": "image", "text": text.strip()}