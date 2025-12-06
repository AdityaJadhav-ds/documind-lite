# context_builder.py — build LLM-ready context with citations

from __future__ import annotations
from typing import List, Dict, Any


def build_context_from_docs(
    docs: List[Dict[str, Any]],
    max_chars: int = 6000,
) -> str:
    """
    Turn ranked docs into a single context string with basic citations.

    Each doc formatted like:

    === DOCUMENT 1 ===
    ID: <id>
    TYPE: <doc_type>
    SOURCE: <source_file>

    TEXT:
    <snippet>

    """
    parts: List[str] = []
    current_len = 0

    for idx, d in enumerate(docs, start=1):
        doc_id = d["id"]
        meta = d.get("metadata", {}) or {}
        doc_type = meta.get("doc_type", "unknown")
        source_file = meta.get("source_file", "")

        header = (
            f"=== DOCUMENT {idx} ===\n"
            f"ID: {doc_id}\n"
            f"TYPE: {doc_type}\n"
            f"SOURCE: {source_file}\n\n"
            f"TEXT:\n"
        )

        text = d.get("text", "") or ""
        snippet = text.strip()

        chunk = header + snippet + "\n\n"
        if current_len + len(chunk) > max_chars:
            # if header alone fits but text doesn't, truncate text
            remaining = max_chars - current_len
            if remaining > len(header) + 100:  # ensure some text
                allowed_text = snippet[: remaining - len(header)]
                chunk = header + allowed_text + "\n\n"
                parts.append(chunk)
            break

        parts.append(chunk)
        current_len += len(chunk)

    return "".join(parts)
