# ocr.py
import easyocr
from typing import List

reader = easyocr.Reader(['en', 'ar'], gpu=False)

def extract_text(image_path: str) -> List[str]:
    """
    Extract text from an image using EasyOCR.
    Returns a list of lines found in the image.
    """
    try:
        results = reader.readtext(image_path, detail=0)
        cleaned_text = [line.strip() for line in results if line.strip()]
        return cleaned_text
    except Exception as e:
        print(f"[ERROR] Failed to extract text: {e}")
        return []

