# ocr/run_ocr_pdfs.py

import os
import json
from pathlib import Path

from pdf2image import convert_from_path
import pytesseract

# 🧠 1) Tell pytesseract where Tesseract is installed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 🧠 2) Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent      # .../documind-lite/
SRC_DIR = BASE_DIR / "data" / "samples"                # PDFs live under here (including subfolders)
OUT_DIR = BASE_DIR / "data" / "ocr_texts"              # We will put .txt + ocr_index.json here
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 🧠 3) Poppler bin path (for pdf2image) – YOUR PATH
POPPLER_PATH = r"Z:\Documind project\Release-25.11.0-0\poppler-25.11.0\Library\bin"

print(f"Using POPPLER_PATH: {POPPLER_PATH}")
print(f"Looking for PDFs under: {SRC_DIR}")

index: dict[str, str] = {}

# 🔁 4) Walk through all subfolders and find *.pdf
pdf_files = list(SRC_DIR.rglob("*.pdf"))
if not pdf_files:
    print("⚠ No PDF files found under data/samples. Check your folder structure.")
else:
    print(f"Found {len(pdf_files)} PDF(s). Starting OCR...")

for pdf_path in pdf_files:
    pdf_path = pdf_path.resolve()
    print(f"\nProcessing: {pdf_path}")

    # 5) Convert PDF pages to images
    pages = convert_from_path(str(pdf_path), poppler_path=POPPLER_PATH)

    # 6) Run OCR on each page
    text_chunks = []
    for i, page in enumerate(pages, start=1):
        print(f"  - OCR page {i}/{len(pages)}")
        page_text = pytesseract.image_to_string(page)
        text_chunks.append(page_text)

    full_text = "\n".join(text_chunks)

    # 7) Save text to file named after the PDF
    txt_name = pdf_path.stem + ".txt"          # e.g., 0.pdf -> 0.txt
    txt_path = OUT_DIR / txt_name
    txt_path.write_text(full_text, encoding="utf-8")

    # 8) Store preview in index (first 400 characters)
    index[pdf_path.name] = full_text[:400]

# 9) Save index as JSON
index_path = OUT_DIR / "ocr_index.json"
index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")

print("\n✅ OCR complete.")
print(f"Text files saved in: {OUT_DIR}")
print(f"OCR index saved as:  {index_path}")
