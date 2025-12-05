# ocr_engine.py — OCR helpers for DocuMind Lite
# Supports multiple backends: "tesseract" (default) and "paddle" (optional).

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Tuple, List

import streamlit as st

from invoice_extractor import extract_invoice_fields, upsert_invoice_row
from doc_classifier import classify_doc
from invoice_extractor import extract_invoice_fields, upsert_invoice_row
from resume_contract_extractor import (
    extract_resume_fields,
    upsert_resume_row,
    extract_contract_fields,
    upsert_contract_row,
)

# ============================================================
# CONFIG — choose OCR backend: "tesseract" or "paddle"
# ============================================================

# You can switch this later or expose it in the UI if you want
OCR_BACKEND = "tesseract"  # change to "paddle" if you have PaddleOCR installed


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """Basic cleaning for noisy OCR text."""
    import re

    if not text:
        return ""
    text = text.replace("\x0c", " ")
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# TESSERACT BACKEND
# ============================================================

def _ocr_pdf_tesseract(file_bytes: bytes) -> Tuple[bool, str]:
    """
    Run OCR on a PDF (all pages) using PyMuPDF (pymupdf) + Tesseract.
    This does NOT require poppler.
    """
    try:
        import fitz  # PyMuPDF
        from PIL import Image
        import pytesseract
    except ImportError:
        return (
            False,
            "Tesseract OCR dependencies missing. Install: "
            "`pip install pymupdf pillow pytesseract` and make sure Tesseract is installed.",
        )

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        return False, f"Error opening PDF with PyMuPDF (Tesseract backend): {e}"

    texts: List[str] = []

    try:
        for page in doc:
            # higher DPI = better OCR but slower
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            txt = pytesseract.image_to_string(img)
            texts.append(txt)
    except Exception as e:
        return False, f"Error during PDF OCR with Tesseract: {e}"
    finally:
        doc.close()

    full_text = "\n\n".join(texts)
    return True, full_text


def _ocr_image_tesseract(file_bytes: bytes) -> Tuple[bool, str]:
    """Run OCR on an image (PNG/JPG) using Tesseract."""
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        return (
            False,
            "Tesseract OCR dependencies missing. Install: `pip install pillow pytesseract` "
            "and make sure Tesseract is installed.",
        )

    try:
        img = Image.open(BytesIO(file_bytes))
        txt = pytesseract.image_to_string(img)
    except Exception as e:
        return False, f"Error during image OCR with Tesseract: {e}"

    return True, txt


# ============================================================
# PADDLEOCR BACKEND (OPTIONAL)
# ============================================================

def _get_paddle_ocr():
    """Lazy-load PaddleOCR; return None if not installed."""
    try:
        from paddleocr import PaddleOCR

        # CPU mode, English language
        ocr = PaddleOCR(use_angle_cls=True, lang="en")
        return ocr
    except Exception as e:
        st.warning(
            f"PaddleOCR backend selected, but PaddleOCR could not be imported: {e}. "
            "Falling back to Tesseract."
        )
        return None


def _ocr_pdf_paddle(file_bytes: bytes) -> Tuple[bool, str]:
    """
    Run OCR on PDF using PaddleOCR.
    If PaddleOCR is unavailable, caller should fall back to Tesseract.
    """
    try:
        import fitz  # PyMuPDF
        import numpy as np
    except ImportError:
        return (
            False,
            "PyMuPDF + numpy required for PaddleOCR PDF. Install: `pip install pymupdf numpy`.",
        )

    ocr = _get_paddle_ocr()
    if ocr is None:
        return False, "PaddleOCR not available."

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        return False, f"Error opening PDF with PyMuPDF (Paddle backend): {e}"

    texts: List[str] = []

    try:
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            result = ocr.ocr(img_array, cls=True)
            page_text_parts: List[str] = []
            for line in result:
                for box, (txt, _score) in line:
                    page_text_parts.append(txt)
            texts.append("\n".join(page_text_parts))
    except Exception as e:
        return False, f"Error during PDF OCR with PaddleOCR: {e}"
    finally:
        doc.close()

    full_text = "\n\n".join(texts)
    return True, full_text


def _ocr_image_paddle(file_bytes: bytes) -> Tuple[bool, str]:
    """Run OCR on image using PaddleOCR."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return (
            False,
            "Pillow + numpy required for PaddleOCR image OCR. Install: `pip install pillow numpy`.",
        )

    ocr = _get_paddle_ocr()
    if ocr is None:
        return False, "PaddleOCR not available."

    try:
        img = Image.open(BytesIO(file_bytes)).convert("RGB")
        img_array = np.array(img)
        result = ocr.ocr(img_array, cls=True)
        parts: List[str] = []
        for line in result:
            for box, (txt, _score) in line:
                parts.append(txt)
        return True, "\n".join(parts)
    except Exception as e:
        return False, f"Error during image OCR with PaddleOCR: {e}"


# ============================================================
# PUBLIC OCR API (used by app.py and other modules)
# ============================================================

def ocr_pdf(file_bytes: bytes) -> Tuple[bool, str]:
    """
    Public OCR entrypoint for PDFs, using the selected backend.
    Tries PaddleOCR first (if configured), then falls back to Tesseract.
    """
    if OCR_BACKEND == "paddle":
        ok, text = _ocr_pdf_paddle(file_bytes)
        if ok:
            st.info("Using PaddleOCR (PDF backend).")
            return True, text
        st.info("PaddleOCR PDF failed or unavailable — falling back to Tesseract.")
    # Default + fallback
    ok, text = _ocr_pdf_tesseract(file_bytes)
    if ok:
        st.info("Using Tesseract (PDF backend).")
    return ok, text


def ocr_image(file_bytes: bytes) -> Tuple[bool, str]:
    """
    Public OCR entrypoint for images, using the selected backend.
    Tries PaddleOCR first (if configured), then falls back to Tesseract.
    """
    if OCR_BACKEND == "paddle":
        ok, text = _ocr_image_paddle(file_bytes)
        if ok:
            st.info("Using PaddleOCR (image backend).")
            return True, text
        st.info("PaddleOCR image failed or unavailable — falling back to Tesseract.")
    # Default + fallback
    ok, text = _ocr_image_tesseract(file_bytes)
    if ok:
        st.info("Using Tesseract (image backend).")
    return ok, text


# ============================================================
# PUBLIC: OCR + INDEX PIPELINE
# ============================================================

def ocr_and_index_upload(
    uploaded_file, doc_type: str, collection
) -> Tuple[bool, str]:
    """
    OCR the uploaded file and add it to the Chroma collection.

    - Uses the configured OCR backend (Tesseract or Paddle).
    - If doc_type == "auto-detect", we run a simple classifier on the OCR text.
    - If final type is 'invoice' / 'resume' / 'contract', we also extract structured
      fields and upsert into the corresponding CSV.
    - Returns (success, message).
    """
    from uuid import uuid4

    ext = Path(uploaded_file.name).suffix.lower()
    file_bytes = uploaded_file.getvalue()

    # 1) OCR by file extension
    if ext == ".pdf":
        ok, text = ocr_pdf(file_bytes)
    elif ext in {".png", ".jpg", ".jpeg"}:
        ok, text = ocr_image(file_bytes)
    else:
        return False, f"Unsupported file type: {ext}. Use PDF / PNG / JPG."

    if not ok:
        # text contains error message in that case
        return False, text

    # 2) Clean text
    cleaned = clean_text(text)
    if len(cleaned) < 40:
        return False, "OCR text too short or empty. Check document quality."

    # 3) Auto-detect doc type (if requested)
    final_doc_type = doc_type
    if doc_type == "auto-detect":
        guessed = classify_doc(cleaned, filename=uploaded_file.name)
        final_doc_type = guessed or "other"

    # 4) Build metadata
    new_id = f"app_{uuid4().hex}"
    meta = {
        "doc_type": final_doc_type,
        "source_file": uploaded_file.name,
        "rel_path": f"uploads/{uploaded_file.name}",
        "text_len": len(cleaned),
        "uploaded_via": "app_upload",
        "uploaded_at": datetime.utcnow().isoformat(),
    }

    # 5) Index in Chroma
    try:
        collection.add(
            ids=[new_id],
            documents=[cleaned],
            metadatas=[meta],
        )
    except Exception as e:
        return False, f"Failed to index document in Chroma: {e}"

    # 6) Structured extraction per doc_type
    try:
        if final_doc_type == "invoice":
            row = extract_invoice_fields(
                text=cleaned, filename=uploaded_file.name, doc_id=new_id
            )
            upsert_invoice_row(row)

        elif final_doc_type == "resume":
            row = extract_resume_fields(
                text=cleaned, filename=uploaded_file.name, doc_id=new_id
            )
            upsert_resume_row(row)

        elif final_doc_type == "contract":
            row = extract_contract_fields(
                text=cleaned, filename=uploaded_file.name, doc_id=new_id
            )
            upsert_contract_row(row)
    except Exception as e:
        # Don't break the whole upload if extraction fails
        st.warning(f"Structured extraction failed for {final_doc_type}: {e}")

    # 7) User-facing message
    base_msg = f"Indexed `{uploaded_file.name}` as `{new_id}` (type: {final_doc_type})."
    if doc_type == "auto-detect":
        base_msg = (
            f"Indexed `{uploaded_file.name}` as `{new_id}` "
            f"(auto-detected type: {final_doc_type})."
        )

    return True, base_msg
