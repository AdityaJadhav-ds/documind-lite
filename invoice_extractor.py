# invoice_extractor.py — Rule-based invoice field extraction for DocuMind Lite

from __future__ import annotations
from structured_db import upsert_invoice_db

import re
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

# Path for invoices CSV (must match app.py logic)
BASE_DIR = Path(__file__).resolve().parent
STRUCTURED_DIR = BASE_DIR / "data" / "structured"
STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)
INVOICES_CSV = STRUCTURED_DIR / "invoices.csv"


def extract_invoice_fields(
    text: str, filename: str, doc_id: str
) -> Dict[str, Any]:
    """
    Very simple heuristic extractor for invoices.

    Tries to extract:
      - invoice_number
      - invoice_date
      - total_amount
      - currency
      - vendor_name
      - client_name

    Returns a dict with safe defaults. Missing fields are set to "".
    """
    t = text or ""
    lower = t.lower()

    # Remove multiple spaces for easier regex
    t_clean = re.sub(r"\s+", " ", t)

    # ---------- invoice number ----------
    invoice_number = ""
    # common patterns: Invoice No: 12345, Invoice #: 12345, INVOICE 12345
    patterns_invoice = [
        r"invoice\s*(no\.?|number|#)\s*[:\-]?\s*([A-Za-z0-9\-\/]+)",
        r"inv\s*(no\.?|number|#)\s*[:\-]?\s*([A-Za-z0-9\-\/]+)",
        r"invoice\s+([A-Za-z0-9\-\/]+)",
    ]
    for pat in patterns_invoice:
        m = re.search(pat, t_clean, flags=re.IGNORECASE)
        if m:
            invoice_number = m.group(len(m.groups()))
            break

    # ---------- invoice date ----------
    invoice_date = ""
    # naive date patterns: 2025-01-31, 31/01/2025, 31-01-2025, Jan 5 2025
    date_patterns = [
        r"\b(\d{4}-\d{2}-\d{2})\b",
        r"\b(\d{2}/\d{2}/\d{4})\b",
        r"\b(\d{2}-\d{2}-\d{4})\b",
        r"\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b",  # 5 Jan 2025
        r"\b([A-Za-z]{3,9}\s+\d{1,2},\s*\d{4})\b",  # Jan 5, 2025
    ]
    for pat in date_patterns:
        m = re.search(pat, t_clean)
        if m:
            invoice_date = m.group(1)
            break

    # ---------- total amount ----------
    total_amount = ""
    currency = ""

    # look for lines containing 'total', 'grand total', 'amount due'
    total_patterns = [
        r"(grand\s+total|total\s+amount|amount\s+due|total)\s*[:\-]?\s*([\$€£₹]?\s?[\d,]+\.?\d*)",
    ]
    for pat in total_patterns:
        m = re.search(pat, t_clean, flags=re.IGNORECASE)
        if m:
            raw = m.group(2).strip()
            total_amount = raw
            # currency guess
            if raw.startswith("$"):
                currency = "USD"
            elif raw.startswith("€"):
                currency = "EUR"
            elif raw.startswith("£"):
                currency = "GBP"
            elif "₹" in raw or "rs" in lower or "inr" in lower:
                currency = "INR"
            break

    # ---------- vendor / client names (very rough) ----------
    vendor_name = ""
    client_name = ""

    # Simple heuristics: look for "Bill To", "Billed To", "Ship To"
    bill_to_match = re.search(
        r"(bill to|billed to|invoice to)\s*[:\-]?\s*(.+?)(?:\s{2,}|\n)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if bill_to_match:
        client_name = bill_to_match.group(2).strip().split("\n")[0][:120]

    from_match = re.search(
        r"(from|seller|vendor)\s*[:\-]?\s*(.+?)(?:\s{2,}|\n)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if from_match:
        vendor_name = from_match.group(2).strip().split("\n")[0][:120]

    # Fallbacks
    if not vendor_name:
        # maybe the first non-empty line is vendor
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            vendor_name = lines[0][:120]

    data = {
        "doc_id": doc_id,
        "source_file": filename,
        "invoice_number": invoice_number,
        "invoice_date": invoice_date,
        "total_amount": total_amount,
        "currency": currency,
        "vendor_name": vendor_name,
        "client_name": client_name,
    }

    return data


def _load_invoices_df() -> pd.DataFrame:
    """Load existing invoices.csv or return empty DataFrame with proper columns."""
    if not INVOICES_CSV.exists():
        return pd.DataFrame(
            columns=[
                "doc_id",
                "source_file",
                "invoice_number",
                "invoice_date",
                "total_amount",
                "currency",
                "vendor_name",
                "client_name",
            ]
        )

    try:
        return pd.read_csv(INVOICES_CSV)
    except Exception:
        # if corrupted, start fresh
        return pd.DataFrame(
            columns=[
                "doc_id",
                "source_file",
                "invoice_number",
                "invoice_date",
                "total_amount",
                "currency",
                "vendor_name",
                "client_name",
            ]
        )


def upsert_invoice_row(fields: Dict[str, Any]) -> None:
    """
    Insert or update a single invoice row in invoices.csv based on doc_id.
    If doc_id exists, row is replaced; otherwise it's appended.
    """
    df = _load_invoices_df()

    doc_id = fields.get("doc_id", "")
    if not doc_id:
        # nothing we can do without a doc_id
        return

    # Remove any existing row with same doc_id
    if not df.empty and "doc_id" in df.columns:
        df = df[df["doc_id"] != doc_id]

    # Append new row
    df = pd.concat([df, pd.DataFrame([fields])], ignore_index=True)

    # Save back
    df.to_csv(INVOICES_CSV, index=False)
