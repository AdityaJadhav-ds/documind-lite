# extraction/batch_extract_invoices.py

import re
import json
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OCR_DIR = BASE_DIR / "data" / "ocr_texts"
OUT_DIR = BASE_DIR / "data" / "structured"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Regex patterns (same logic as single-file script) ----------

invoice_pattern = re.compile(
    r"(Invoice\s*(No\.?|#|ID)?\s*[:\-]?\s*|#\s*)([A-Z0-9\-\/]+)",
    re.I,
)

month_name_date_pattern = re.compile(
    r"(Invoice\s*Date|Date\s*of\s*Issue|Issue\s*Date|Billing\s*Date|Date)\s*"
    r"[:\-]?\s*"
    r"([A-Za-z]{3,9}\s+\d{1,2}\s+\d{4})",   # e.g. Mar 30 2012
    re.I,
)

numeric_date_pattern = re.compile(
    r"([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4})",
    re.I,
)

total_pattern = re.compile(
    r"(Grand\s*Total|Total\s*Amount|Amount\s*Due|Invoice\s*Total|Total)\s*"
    r"[:\-]?\s*\$?\s*([0-9][0-9,]*\.?[0-9]{0,2})",
    re.I,
)


def extract_fields(text: str, filename: str) -> dict:
    # Invoice number
    inv_match = invoice_pattern.search(text)

    # Date: month-name style first (Date: Mar 30 2012)
    invoice_date = None
    month_match = month_name_date_pattern.search(text)
    if month_match:
        invoice_date = month_match.group(2).strip()
    else:
        # fallback: any numeric date (e.g., 30/03/2012)
        num_match = numeric_date_pattern.search(text)
        if num_match:
            invoice_date = num_match.group(1).strip()

    # Total amount
    tot_match = total_pattern.search(text)

    return {
        "doc_id": Path(filename).stem,
        "source_file": filename,
        "invoice_number": inv_match.group(3) if inv_match else None,
        "invoice_date": invoice_date,
        "total_amount": tot_match.group(2) if tot_match else None,
    }


def main():
    records = []

    # Loop through all OCR txt files
    for txt_file in OCR_DIR.glob("*.txt"):
        # Process only invoices for now
        if not txt_file.name.lower().startswith("invoice"):
            continue

        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        record = extract_fields(text, txt_file.name)
        records.append(record)
        print(f"Processed: {txt_file.name} -> {record}")

    if not records:
        print("No invoice TXT files found or no records extracted.")
        return

    # Save JSON
    json_path = OUT_DIR / "invoices.json"
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    # Save CSV
    csv_path = OUT_DIR / "invoices.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print("\n✅ Saved invoice records to:")
    print(" -", json_path)
    print(" -", csv_path)
    print(f"Total invoices processed: {len(records)}")


if __name__ == "__main__":
    main()
