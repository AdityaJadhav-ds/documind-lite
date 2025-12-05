# doc_classifier.py — Tiny heuristic doc-type classifier for DocuMind Lite

from __future__ import annotations

from typing import Literal

DocType = Literal["invoice", "resume", "contract", "other"]


def classify_doc(text: str, filename: str = "") -> DocType:
    """
    Very simple rule-based classifier.

    Uses filename + keywords in text to guess:
      - invoice
      - resume
      - contract
      - other
    """
    if text is None:
        text = ""
    if filename is None:
        filename = ""

    t = text.lower()
    f = filename.lower()

    # --- invoice-like signals ---
    invoice_keywords = [
        "invoice",
        "bill to",
        "amount due",
        "subtotal",
        "vat",
        "gst",
        "balance due",
        "purchase order",
        "po number",
    ]

    # --- resume-like signals ---
    resume_keywords = [
        "curriculum vitae",
        "resume",
        "summary",
        "experience",
        "work experience",
        "professional experience",
        "skills",
        "education",
        "projects",
        "technologies",
    ]

    # --- contract-like signals ---
    contract_keywords = [
        "agreement",
        "contract",
        "party of the first part",
        "party of the second part",
        "hereinafter referred to as",
        "terms and conditions",
        "governing law",
        "indemnify",
        "termination",
        "confidentiality",
    ]

    # filename hints
    if "invoice" in f:
        return "invoice"
    if "resume" in f or "cv" in f:
        return "resume"
    if "contract" in f or "agreement" in f:
        return "contract"

    # text keyword counts
    inv_score = sum(kw in t for kw in invoice_keywords)
    res_score = sum(kw in t for kw in resume_keywords)
    con_score = sum(kw in t for kw in contract_keywords)

    # pick the highest score if it's clearly above others
    scores = {
        "invoice": inv_score,
        "resume": res_score,
        "contract": con_score,
    }
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    # if everything is 0 or very close, call it "other"
    non_zero = [v for v in scores.values() if v > 0]
    if not non_zero:
        return "other"

    # if best score is only 1 and others are 1 as well, unsure → other
    if best_score <= 1 and sum(1 for v in scores.values() if v == best_score) > 1:
        return "other"

    return best_type
