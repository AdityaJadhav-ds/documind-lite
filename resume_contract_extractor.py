# resume_contract_extractor.py — Resume & Contract field extraction for DocuMind Lite
# -------------------------------------------------------------------------------
# This module contains small, production-friendly, rule-based extractors for:
#   • Resumes   → data/structured/resumes.csv
#   • Contracts → data/structured/contracts.csv
#
# Design goals:
#   - No heavy external NLP libraries (just regex + heuristics).
#   - Self-contained: creates folders / CSVs if they don't exist.
#   - "Upsert" behaviour: same doc_id overwrites previous row.
#   - Safe: extraction failures never crash the app.

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from structured_db import upsert_resume_db, upsert_contract_db

import re
import pandas as pd

# =====================================================================
# PATHS
# =====================================================================

BASE_DIR = Path(__file__).resolve().parent
STRUCTURED_DIR = BASE_DIR / "data" / "structured"
STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)

RESUMES_CSV_PATH = STRUCTURED_DIR / "resumes.csv"
CONTRACTS_CSV_PATH = STRUCTURED_DIR / "contracts.csv"


# =====================================================================
# ===========================  RESUMES  ===============================
# =====================================================================

@dataclass
class ResumeFields:
    """Structured representation of basic resume information."""

    doc_id: str
    source_file: str

    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    headline: Optional[str] = None  # e.g. "Data Scientist"
    total_experience_years: Optional[float] = None
    skills: Optional[str] = None  # comma-separated string
    education_summary: Optional[str] = None

    raw_text_snippet: Optional[str] = None  # first 500 chars for debugging


# ---------- Resume helper functions ----------


def _extract_email(text: str) -> Optional[str]:
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return m.group(0) if m else None


def _extract_phone(text: str) -> Optional[str]:
    # Very loose phone pattern: picks first phone-like sequence
    m = re.search(
        r"(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}",
        text,
    )
    return m.group(0) if m else None


def _extract_name(text: str, filename: str) -> Optional[str]:
    """
    Heuristic:
      1. Use filename (e.g. 'aditya-jadhav-resume.pdf' -> 'Aditya Jadhav').
      2. Else, first line with 2–4 words.
    """
    base = Path(filename).stem
    base = re.sub(r"(?i)\b(resume|cv|curriculum vitae)\b", "", base)
    base_clean = base.replace("_", " ").replace("-", " ").strip()
    if base_clean and 2 <= len(base_clean.split()) <= 4:
        return base_clean.title()

    # Fallback: first non-empty line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None
    first_line = lines[0]
    if 2 <= len(first_line.split()) <= 6:
        return first_line.strip()
    return None


def _extract_skills(text: str) -> Optional[str]:
    """
    Look for 'Skills' section and capture 1–3 lines around it.
    Returns a single comma-separated string.
    """
    lines = [l.strip() for l in text.splitlines()]
    collected: List[str] = []

    for i, line in enumerate(lines):
        lower = line.lower()
        if lower.startswith("skills") or "skills:" in lower:
            block = [line]
            if i + 1 < len(lines):
                block.append(lines[i + 1])
            if i + 2 < len(lines):
                block.append(lines[i + 2])
            collected = block
            break

    if not collected:
        return None

    joined = " ".join(collected)
    joined = re.sub(r"(?i)skills\s*[:\-]\s*", "", joined)
    # Normalize delimiters to commas
    joined = joined.replace("•", ",").replace("|", ",").replace(";", ",")
    joined = re.sub(r"\s*,\s*", ", ", joined)
    return joined.strip(" ,") or None


def _extract_experience_years(text: str) -> Optional[float]:
    """
    Very rough extraction of 'X years of experience'.
    """
    m = re.search(r"(\d+)\s*\+?\s*years? of experience", text.lower())
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _extract_education_summary(text: str) -> Optional[str]:
    """
    Capture lines under an 'Education' heading until a blank line / new section.
    """
    lines = [l.strip() for l in text.splitlines()]
    edu_lines: List[str] = []
    capturing = False

    for line in lines:
        lower = line.lower()
        if "education" in lower:
            capturing = True
            continue
        if capturing:
            if not line:
                break
            # stop if new obvious section header
            if re.match(r"(?i)^(projects|experience|skills|work experience)\b", lower):
                break
            edu_lines.append(line)

    if not edu_lines:
        return None

    return " | ".join(edu_lines)


def _extract_headline(text: str) -> Optional[str]:
    """
    Simple heuristic: second non-empty line, if short, is often a role/headline.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) >= 2 and 1 <= len(lines[1].split()) <= 8:
        return lines[1]
    return None


# ---------- Resume public API ----------


def extract_resume_fields(text: str, filename: str, doc_id: str) -> Dict[str, object]:
    """
    Extract key resume fields from OCR'd text.
    Returns a dict ready to be turned into a DataFrame row.
    """
    snippet = text[:500] if text else ""

    email = _extract_email(text)
    phone = _extract_phone(text)
    full_name = _extract_name(text, filename)
    skills = _extract_skills(text)
    exp_years = _extract_experience_years(text)
    edu_summary = _extract_education_summary(text)
    headline = _extract_headline(text)

    fields = ResumeFields(
        doc_id=doc_id,
        source_file=filename,
        full_name=full_name,
        email=email,
        phone=phone,
        headline=headline,
        total_experience_years=exp_years,
        skills=skills,
        education_summary=edu_summary,
        raw_text_snippet=snippet,
    )

    return asdict(fields)


def upsert_resume_row(row: Dict[str, object]) -> None:
    """
    Insert or update a row in resumes.csv based on doc_id.
    Creates the file if it doesn't exist.
    """
    RESUMES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    if RESUMES_CSV_PATH.exists():
        df = pd.read_csv(RESUMES_CSV_PATH)
    else:
        df = pd.DataFrame()

    doc_id = row.get("doc_id")

    if "doc_id" not in df.columns:
        df = pd.DataFrame([row])
    else:
        mask = df["doc_id"] == doc_id
        if mask.any():
            df.loc[mask, :] = pd.DataFrame([row]).values
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(RESUMES_CSV_PATH, index=False)


# =====================================================================
# ==========================  CONTRACTS  ==============================
# =====================================================================

@dataclass
class ContractFields:
    """Structured representation of basic contract information."""

    doc_id: str
    source_file: str

    party_a: Optional[str] = None
    party_b: Optional[str] = None
    effective_date: Optional[str] = None
    end_date: Optional[str] = None
    governing_law: Optional[str] = None
    contract_title: Optional[str] = None

    raw_text_snippet: Optional[str] = None


# ---------- Contract helper functions ----------


def _extract_parties(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Very rough heuristic:
      - Look for 'between X and Y.'
      - or 'This Agreement is between X and Y'.
    """
    m = re.search(
        r"between\s+(.+?)\s+and\s+(.+?)[\.,\n]",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        party_a = m.group(1).strip()
        party_b = m.group(2).strip()
        return party_a, party_b
    return None, None


def _extract_effective_date_raw(text: str) -> Optional[str]:
    """
    Grab the phrase after 'effective as of' / 'effective date' up to ~40 chars.
    We don't fully parse dates here; we just keep a readable snippet.
    """
    m = re.search(
        r"effective (as of|date)\s*[:\-]?\s*(.+)",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    tail = m.group(2)
    # up to first newline or comma
    tail = tail.split("\n")[0]
    tail = tail.split(",")[0]
    return tail.strip() or None


def _extract_governing_law(text: str) -> Optional[str]:
    """
    Look for 'governed by the laws of X'.
    """
    m = re.search(
        r"governed by the laws of\s+([A-Za-z\s]+)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return None


def _extract_contract_title(text: str, filename: str) -> Optional[str]:
    """
    Guess contract title:
      1. From filename (e.g. 'Service_Agreement_2025.pdf' -> 'Service Agreement 2025')
      2. Else, first all-caps-ish line.
    """
    base = Path(filename).stem.replace("_", " ").replace("-", " ").strip()
    if base and len(base.split()) <= 10:
        return base.title()

    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        # all caps or mostly caps
        letters = re.sub(r"[^A-Za-z]", "", s)
        if letters and letters.upper() == letters and 3 <= len(s) <= 80:
            return s.title()
    return None


# ---------- Contract public API ----------


def extract_contract_fields(text: str, filename: str, doc_id: str) -> Dict[str, object]:
    """
    Extract key contract fields from OCR'd text.
    Returns a dict ready to be turned into a DataFrame row.
    """
    snippet = text[:500] if text else ""

    party_a, party_b = _extract_parties(text)
    effective_date = _extract_effective_date_raw(text)
    governing_law = _extract_governing_law(text)
    title = _extract_contract_title(text, filename)

    fields = ContractFields(
        doc_id=doc_id,
        source_file=filename,
        party_a=party_a,
        party_b=party_b,
        effective_date=effective_date,
        end_date=None,  # could be extracted in a later upgrade
        governing_law=governing_law,
        contract_title=title,
        raw_text_snippet=snippet,
    )

    return asdict(fields)


def upsert_contract_row(row: Dict[str, object]) -> None:
    """
    Insert or update a row in contracts.csv based on doc_id.
    Creates the file if it doesn't exist.
    """
    CONTRACTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    if CONTRACTS_CSV_PATH.exists():
        df = pd.read_csv(CONTRACTS_CSV_PATH)
    else:
        df = pd.DataFrame()

    doc_id = row.get("doc_id")

    if "doc_id" not in df.columns:
        df = pd.DataFrame([row])
    else:
        mask = df["doc_id"] == doc_id
        if mask.any():
            df.loc[mask, :] = pd.DataFrame([row]).values
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(CONTRACTS_CSV_PATH, index=False)
