# extraction/extract_invoice_fields.py

import re
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OCR_DIR = BASE_DIR / "data" / "ocr_texts"

# 👇 CHANGE THIS to test different invoices
TEST_FILE = "invoice_Aaron Smayling_35876.txt"
txt_path = OCR_DIR / TEST_FILE

text = txt_path.read_text(encoding="utf-8", errors="ignore")

# ---------- 1) Invoice number ----------
invoice_pattern = re.compile(
    r"(Invoice\s*(No\.?|#|ID)?\s*[:\-]?\s*|#\s*)([A-Z0-9\-\/]+)",
    re.I,
)
inv_match = invoice_pattern.search(text)

# ---------- 2) Invoice date (supports: Date: Mar 30 2012) ----------

# Month names/abbreviations: Jan, January, Feb, February, ...
month_name_date_pattern = re.compile(
    r"(Invoice\s*Date|Date\s*of\s*Issue|Issue\s*Date|Billing\s*Date|Date)\s*"
    r"[:\-]?\s*"
    r"([A-Za-z]{3,9}\s+\d{1,2}\s+\d{4})",   # e.g. Mar 30 2012
    re.I,
)

# Numeric formats (fallback): 30/03/2012, 30-03-12, etc.
numeric_date_pattern = re.compile(
    r"([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4})",
    re.I,
)

invoice_date = None

labelled_month_match = month_name_date_pattern.search(text)
if labelled_month_match:
    invoice_date = labelled_month_match.group(2).strip()
else:
    # fallback: any numeric date in the text
    any_date_match = numeric_date_pattern.search(text)
    if any_date_match:
        invoice_date = any_date_match.group(1).strip()

# ---------- 3) Total amount ----------
total_pattern = re.compile(
    r"(Grand\s*Total|Total\s*Amount|Amount\s*Due|Invoice\s*Total|Total)\s*"
    r"[:\-]?\s*\$?\s*([0-9][0-9,]*\.?[0-9]{0,2})",
    re.I,
)
tot_match = total_pattern.search(text)

data = {
    "source_file": TEST_FILE,
    "invoice_number": inv_match.group(3) if inv_match else None,
    "invoice_date": invoice_date,
    "total_amount": tot_match.group(2) if tot_match else None,
}

print(json.dumps(data, indent=2))
