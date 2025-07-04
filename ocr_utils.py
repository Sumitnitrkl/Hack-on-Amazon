import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from langchain.docstore.document import Document

# Set Tesseract path (update if different on your system)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_image(image_path):
    """Extract text from an image file using Tesseract OCR."""
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return Document(page_content=text, metadata={"source": image_path})
    except Exception as e:
        print(f"[ERROR] Failed to process image {image_path}: {e}")
        return None

def extract_text_from_scanned_pdf(pdf_path):
    """Extract text from a scanned PDF using Tesseract OCR."""
    try:
        pages = convert_from_path(pdf_path)
        text = ""
        for i, page in enumerate(pages):
            page_text = pytesseract.image_to_string(page)
            text += f"\nPage {i+1}:\n{page_text}"
        return Document(page_content=text, metadata={"source": pdf_path})
    except Exception as e:
        print(f"[ERROR] Failed to process PDF {pdf_path}: {e}")
        return None
