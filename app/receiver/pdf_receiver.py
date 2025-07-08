import os, tempfile, requests
from typing import Final
from PIL import Image, ImageOps
import pytesseract
import fitz

_TIMEOUT: Final[int] = 30

class PDFReceiver:
    """PDF 링크 → 텍스트(이미지 OCR 포함)"""
    def fetch_and_extract_text(self, url: str) -> str:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as fp:
            fp.write(resp.content)
            pdf_path = fp.name
        try:
            parser = PDFParser()
            elements = parser.read(pdf_path)
            return "\n".join(e for e in elements if e)
        finally:
            os.remove(pdf_path)


class PDFParser:
    def __init__(self, ocr_lang: str = "kor+eng"):
        self.ocr_lang = ocr_lang

    def read(self, pdf_path: str) -> list:
        with fitz.open(pdf_path) as doc:
            texts = []
            for page in doc:
                text = page.get_text("text")
                if len(text.strip()) > 50:
                    texts.append(text)
                else:
                    texts.append(self._ocr_page(page))
        return texts
    
    def _ocr_page(self, page, OCR_DPI = 300) -> str:
        try:
            pix = page.get_pixmap(dpi=OCR_DPI)
            img = pix.pil_image
            gray = ImageOps.grayscale(img)
            bw = gray.point(lambda x: 0 if x < 180 else 255, "1")
            return pytesseract.image_to_string(bw, lang=self.ocr_lang, timeout=10)
        except Exception as e:
            return ""
    