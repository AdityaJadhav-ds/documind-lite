# query_analysis.py — simple query intent classifier for DocuMind Lite

from __future__ import annotations
from typing import Literal, Dict
import re

QueryType = Literal[
    "exact_lookup",     # invoice IDs, filenames, short codes
    "semantic_qa",      # natural language questions
    "count_query",      # how many / number of ...
    "filter_stats",     # total sum, min/max, etc.
    "unknown",
]


def classify_query(query: str) -> Dict[str, str]:
    """
    Classify the user query into coarse types.

    Returns:
        {
          "type": QueryType,
          "reason": str
        }
    """
    q = (query or "").strip().lower()
    if not q:
        return {"type": "unknown", "reason": "Empty query"}

    # Count queries
    # e.g. "how many invoices", "number of resumes", "count of contracts"
    if any(kw in q for kw in ["how many", "number of", "count of", "total invoices", "total resumes"]):
        return {"type": "count_query", "reason": "Detected counting keywords"}

    # Filter/stat queries
    # e.g. "total amount", "sum of", "max amount", "minimum salary"
    if any(kw in q for kw in ["total amount", "sum of", "total value", "maximum", "minimum", "avg", "average"]):
        return {"type": "filter_stats", "reason": "Detected aggregation keywords"}

    # Exact lookup heuristics
    # - Looks like an ID / code: contains dashes/numbers/uppercase letters
    # - Often short length (<= 30 chars)
    id_like = bool(re.search(r"[a-z]{2,}\-\d{2,}", q))  # like INV-2024-01
    has_digits_and_letters = bool(re.search(r"[a-z]", q) and re.search(r"\d", q))
    short_query = len(q) <= 30

    if (id_like or has_digits_and_letters) and short_query:
        return {"type": "exact_lookup", "reason": "Short alphanumeric code-like query"}

    # If it looks like a question (who/what/when/where/how)
    if any(q.startswith(w) for w in ["what", "who", "when", "where", "why", "how"]):
        return {"type": "semantic_qa", "reason": "Question-style query"}

    # If it's a long sentence → semantic
    if len(q.split()) >= 6:
        return {"type": "semantic_qa", "reason": "Long natural language query"}

    # Default: unknown → treat as hybrid search
    return {"type": "unknown", "reason": "Fallback"}
