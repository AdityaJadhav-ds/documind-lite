# keyword_search.py — BM25-style keyword search for DocuMind Lite
# ---------------------------------------------------------------
# - Works with new/old ChromaDB (Streamlit Cloud compatible)
# - No include=["ids"] (ids are returned automatically)
# - Supports doc_type_filter just like vector search
# - Safe: returns [] on no matches / empty query

from __future__ import annotations

from typing import List, Dict, Any
import re
import math


# ===================== TOKENIZATION =====================


def _tokenize(text: str) -> List[str]:
    """Lowercase + keep only alphanumeric tokens."""
    if not text:
        return []
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


# ===================== BM25 INDEX BUILDING =====================


def _build_bm25_index(texts: List[str]) -> Dict[str, Any]:
    """
    Build BM25 index from list of texts.

    Returns a dict with:
        - doc_tokens: List[List[str]]
        - df: Dict[token, document_frequency]
        - avgdl: float (average document length)
        - N: int (number of documents)
    """
    doc_tokens: List[List[str]] = []
    df: Dict[str, int] = {}
    total_len = 0

    for txt in texts:
        tokens = _tokenize(txt)
        doc_tokens.append(tokens)
        total_len += len(tokens)
        unique_tokens = set(tokens)
        for t in unique_tokens:
            df[t] = df.get(t, 0) + 1

    N = len(doc_tokens)
    avgdl = total_len / float(N or 1)

    return {
        "doc_tokens": doc_tokens,
        "df": df,
        "avgdl": avgdl,
        "N": N,
    }


def _bm25_scores(query: str, index: Dict[str, Any]) -> List[float]:
    """Compute BM25-like scores for each document."""
    k1 = 1.5
    b = 0.75

    q_tokens = _tokenize(query)
    if not q_tokens or index["N"] == 0:
        return [0.0] * index["N"]

    doc_tokens: List[List[str]] = index["doc_tokens"]
    df: Dict[str, int] = index["df"]
    N: int = index["N"]
    avgdl: float = index["avgdl"] or 1.0

    scores: List[float] = []

    for tokens in doc_tokens:
        dl = len(tokens) or 1
        score = 0.0

        # term frequency in this doc
        freq: Dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1

        for t in q_tokens:
            ni = df.get(t, 0)
            if ni == 0:
                continue

            fi = freq.get(t, 0)
            if fi == 0:
                continue

            # IDF with standard BM25 smoothing
            idf = math.log(1 + (N - ni + 0.5) / (ni + 0.5))

            denom = fi + k1 * (1 - b + b * dl / avgdl)
            score += idf * (fi * (k1 + 1) / denom)

        scores.append(float(score))

    return scores


# ===================== PUBLIC API =====================


def keyword_search(
    collection,
    query: str,
    top_k: int = 5,
    doc_type_filter: str = "all",
) -> List[Dict[str, Any]]:
    """
    BM25-style keyword search over all Chroma documents.

    Args:
        collection: Chroma collection object.
        query: User's search string.
        top_k: Max number of docs to return.
        doc_type_filter: "all" or specific type ("invoice", "resume", "contract").

    Returns:
        List of dicts:
        [
            {
                "id": str,
                "score": float,
                "doc_type": str,
                "source_file": str,
                "text": str,  # snippet
            },
            ...
        ]
    """
    query = (query or "").strip()
    if not query:
        return []

    # Build get() kwargs compatible with new Chroma
    get_kwargs: Dict[str, Any] = {
        "include": ["documents", "metadatas"],
    }
    dt = (doc_type_filter or "all").lower()
    if dt != "all":
        get_kwargs["where"] = {"doc_type": dt}

    # Chroma always returns "ids" even if you don't ask in include
    results = collection.get(**get_kwargs)

    docs: List[str] = results.get("documents", []) or []
    metas: List[Dict[str, Any]] = results.get("metadatas", []) or []
    ids: List[str] = results.get("ids", []) or []

    if not docs:
        return []

    # Safety: align lengths
    n = min(len(docs), len(metas), len(ids))
    docs = docs[:n]
    metas = metas[:n]
    ids = ids[:n]

    # Build BM25 index & score
    index = _build_bm25_index(docs)
    scores = _bm25_scores(query, index)

    # Rank by score
    ranked = sorted(
        zip(ids, metas, docs, scores),
        key=lambda x: x[3],
        reverse=True,
    )

    # Filter out zero-score docs (no keyword hit)
    ranked = [r for r in ranked if r[3] > 0.0][:top_k]

    results_out: List[Dict[str, Any]] = []
    for doc_id, meta, text, score in ranked:
        snippet = text[:800] + ("..." if len(text) > 800 else "")
        results_out.append(
            {
                "id": str(doc_id),
                "score": round(float(score), 4),
                "doc_type": str(meta.get("doc_type", "unknown")),
                "source_file": str(meta.get("source_file", "")),
                "text": snippet,
            }
        )

    return results_out
