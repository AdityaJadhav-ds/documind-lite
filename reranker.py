# reranker.py — embedding-based reranker for DocuMind Lite

from __future__ import annotations
from typing import List, Dict, Any
import math

from openai import OpenAI


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors (no numpy)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def embedding_rerank(
    query: str,
    docs: List[Dict[str, Any]],
    api_key: str,
    model: str = "text-embedding-3-small",
    max_chars_per_doc: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Rerank docs using OpenAI embedding model:
      - Get embedding for query
      - Get embedding for each doc text
      - Sort by cosine similarity

    Docs must be dicts with at least: {"text": ...}.
    """
    query = (query or "").strip()
    if not query or not docs or not api_key:
        return docs

    client = OpenAI(api_key=api_key)

    # Prepare text inputs (truncate to reduce token cost)
    inputs: List[str] = [query]
    for d in docs:
        txt = (d.get("text") or "").strip()
        if len(txt) > max_chars_per_doc:
            txt = txt[:max_chars_per_doc]
        inputs.append(txt)

    try:
        resp = client.embeddings.create(
            model=model,
            input=inputs,
        )
    except Exception:
        # If embedding API fails, fall back to original order
        return docs

    if not resp.data or len(resp.data) < 2:
        return docs

    query_vec = resp.data[0].embedding
    doc_vecs = [item.embedding for item in resp.data[1:]]

    # Attach rerank_score to docs
    for d, vec in zip(docs, doc_vecs):
        d["rerank_score"] = float(_cosine_similarity(query_vec, vec))

    # Sort by rerank_score (desc), but keep hybrid_score as info
    docs_sorted = sorted(docs, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return docs_sorted
