# ocr_engine.py — OCR + Indexing pipeline for DocuMind Lite
# ---------------------------------------------------------
# - Supports multiple backends: "tesseract" (default) and "paddle" (optional)
# - Saves original uploaded files to ./uploads/
# - Adds documents to ChromaDB with proper rel_path/source_file metadata
# - Runs basic document classification + structured extraction (invoice/resume/contract)

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Tuple, List

import streamlit as st

from doc_classifier import classify_doc
from invoice_extractor import extract_invoice_fields, upsert_invoice_row
from resume_contract_extractor import (
    extract_resume_fields,
    upsert_resume_row,
    extract_contract_fields,
    upsert_contract_row,
)

# ============================================================
# GLOBAL PATHS / CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"

# Ensure uploads directory exists
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Choose OCR backend: "tesseract" or "paddle"
# You can later expose this as a Streamlit selectbox if you want.
OCR_BACKEND = "tesseract"  # or "paddle" if PaddleOCR is installed


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """Basic cleaning for noisy OCR text (remove control chars, collapse whitespace)."""
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
    Does NOT require poppler.
    """
    try:
        import fitz  # PyMuPDF
        from PIL import Image
        import pytesseract
    except ImportError:
        return (
            False,
            "Tesseract OCR dependencies missing. Install:\n"
            "  pip install pymupdf pillow pytesseract\n"
            "and make sure Tesseract is installed on your system.",
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
            "Tesseract OCR dependencies missing. Install:\n"
            "  pip install pillow pytesseract\n"
            "and make sure Tesseract is installed on your system.",
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
    """Lazy-load PaddleOCR; return None if not installed or if import fails."""
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
            "PyMuPDF + numpy required for PaddleOCR PDF.\n"
            "Install: pip install pymupdf numpy",
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
            # pix.samples is raw bytes in RGB format
            import numpy as np  # safe here due to guard above
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
            "Pillow + numpy required for PaddleOCR image OCR.\n"
            "Install: pip install pillow numpy",
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
    uploaded_file,
    doc_type: str,
    collection,
) -> Tuple[bool, str]:
    """
    OCR the uploaded file and add it to the Chroma collection.

    Workflow:
      1. Save original file into ./uploads/<filename>
      2. Run OCR (PDF / image) with configured backend
      3. Clean text
      4. Auto-detect doc type if requested ("auto-detect")
      5. Add to Chroma with metadata (doc_type, source_file, rel_path, etc.)
      6. If type is invoice / resume / contract → extract structured fields and upsert

    Returns:
      (success: bool, message: str)
    """
    from uuid import uuid4

    # --------------------------------------------------------
    # 0) Basic checks
    # --------------------------------------------------------
    if uploaded_file is None:
        return False, "No file uploaded."

    filename = uploaded_file.name
    ext = Path(filename).suffix.lower()
    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        return False, "Uploaded file is empty."

    # --------------------------------------------------------
    # 1) Save original file to ./uploads
    # --------------------------------------------------------
    try:
        safe_name = Path(filename).name  # avoid any path traversal
        save_path = UPLOADS_DIR / safe_name
        with open(save_path, "wb") as f:
            f.write(file_bytes)
    except Exception as e:
        return False, f"Failed to save uploaded file to disk: {e}"

    # rel_path is stored relative to BASE_DIR so Viewer can resolve it
    rel_path = f"uploads/{safe_name}"

    # --------------------------------------------------------
    # 2) OCR by file extension
    # --------------------------------------------------------
    if ext == ".pdf":
        ok, text = ocr_pdf(file_bytes)
    elif ext in {".png", ".jpg", ".jpeg"}:
        ok, text = ocr_image(file_bytes)
    else:
        return False, f"Unsupported file type: {ext}. Use PDF / PNG / JPG."

    if not ok:
        # text contains error message in that case
        return False, text

    # --------------------------------------------------------
    # 3) Clean text
    # --------------------------------------------------------
    cleaned = clean_text(text)
    if len(cleaned) < 40:
        return False, "OCR text too short or empty. Check document quality."

    # --------------------------------------------------------
    # 4) Auto-detect doc type (if requested)
    # --------------------------------------------------------
    final_doc_type = (doc_type or "other").lower()
    if final_doc_type == "auto-detect":
        guessed = classify_doc(cleaned, filename=filename)
        final_doc_type = (guessed or "other").lower()

    # --------------------------------------------------------
    # 5) Build metadata & add to Chroma
    # --------------------------------------------------------
    new_id = f"app_{uuid4().hex}"

    meta = {
        "doc_type": final_doc_type,
        "source_file": safe_name,
        "rel_path": rel_path,  # used by Viewer to find original file
        "text_len": len(cleaned),
        "uploaded_via": "app_upload",
        "uploaded_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }

    try:
        collection.add(
            ids=[new_id],
            documents=[cleaned],
            metadatas=[meta],
        )
    except Exception as e:
        return False, f"Failed to index document in Chroma: {e}"

    # --------------------------------------------------------
    # 6) Structured extraction per doc_type (best-effort)
    # --------------------------------------------------------
    try:
        if final_doc_type == "invoice":
            row = extract_invoice_fields(
                text=cleaned,
                filename=safe_name,
                doc_id=new_id,
            )
            upsert_invoice_row(row)

        elif final_doc_type == "resume":
            row = extract_resume_fields(
                text=cleaned,
                filename=safe_name,
                doc_id=new_id,
            )
            upsert_resume_row(row)

        elif final_doc_type == "contract":
            row = extract_contract_fields(
                text=cleaned,
                filename=safe_name,
                doc_id=new_id,
            )
            upsert_contract_row(row)

    except Exception as e:
        # Don't break the whole upload if extraction fails
        st.warning(f"Structured extraction failed for type '{final_doc_type}': {e}")

    # --------------------------------------------------------
    # 7) User-facing message
    # --------------------------------------------------------
    if doc_type == "auto-detect":
        base_msg = (
            f"Indexed `{filename}` as `{new_id}` "
            f"(auto-detected type: {final_doc_type})."
        )
    else:
        base_msg = (
            f"Indexed `{filename}` as `{new_id}` "
            f"(type: {final_doc_type})."
        )

    base_msg += f"\nOriginal file saved at: `{rel_path}`."

    return True, base_msg
