# keyword_search.py — Simple BM25-style keyword search over Chroma documents
# -------------------------------------------------------------------------
# Goal:
#   Provide a lightweight, dependency-free keyword search layer that works
#   alongside Chroma's vector search. This is NOT a perfect BM25 implementation,
#   but a solid "good enough" scorer for a small doc collection.
#
# Public API:
#   keyword_search(collection, query, top_k=5, doc_type_filter="all") -> List[dict]
#
# Returned doc dict format (matches retrieve_docs in app.py):
#   {
#       "id": str,
#       "source_file": str,
#       "doc_type": str,
#       "text": str,
#       "score": float,
#   }

from __future__ import annotations

import math
import re
from typing import Any, Dict, List

import pandas as pd


def _tokenize(text: str) -> List[str]:
    """Lowercase and split on non-alphanumeric. Very simple tokenizer."""
    if not text:
        return []
    text = text.lower()
    tokens = re.split(r"[^a-z0-9]+", text)
    return [t for t in tokens if t]


def _build_corpus_from_chroma(collection, doc_type_filter: str = "all") -> pd.DataFrame:
    """
    Load all documents (or filtered by doc_type) from Chroma into a DataFrame.
    Columns:
        - doc_id
        - doc_type
        - source_file
        - text
    """
    get_kwargs: Dict[str, Any] = {
        "include": ["documents", "metadatas", "ids"],
    }
    dt = (doc_type_filter or "all").lower()
    if dt != "all":
        get_kwargs["where"] = {"doc_type": dt}

    results = collection.get(**get_kwargs)
    ids = results.get("ids", []) or []
    metas = results.get("metadatas", []) or []
    docs = results.get("documents", []) or []

    rows = []
    for doc_id, meta, text in zip(ids, metas, docs):
        rows.append(
            {
                "doc_id": str(doc_id),
                "doc_type": str(meta.get("doc_type", "other")),
                "source_file": str(meta.get("source_file", "")),
                "text": text or "",
            }
        )

    return pd.DataFrame(rows)


def _build_inverted_index(df: pd.DataFrame):
    """
    Build simple inverted index + stats:
        - tokens per doc
        - doc lengths
        - document frequency per term
    Returns:
        corpus_tokens: List[List[str]]
        doc_ids: List[str]
        doc_types: List[str]
        source_files: List[str]
        df_counts: Dict[str, int]
        avgdl: float
    """
    corpus_tokens: List[List[str]] = []
    doc_ids: List[str] = []
    doc_types: List[str] = []
    source_files: List[str] = []
    df_counts: Dict[str, int] = {}

    for _, row in df.iterrows():
        tokens = _tokenize(row["text"])
        corpus_tokens.append(tokens)
        doc_ids.append(row["doc_id"])
        doc_types.append(row["doc_type"])
        source_files.append(row["source_file"])

        seen_terms = set(tokens)
        for term in seen_terms:
            df_counts[term] = df_counts.get(term, 0) + 1

    if corpus_tokens:
        avgdl = sum(len(toks) for toks in corpus_tokens) / float(len(corpus_tokens))
    else:
        avgdl = 0.0

    return corpus_tokens, doc_ids, doc_types, source_files, df_counts, avgdl


def _bm25_scores_for_query(
    query: str,
    corpus_tokens: List[List[str]],
    df_counts: Dict[str, int],
    avgdl: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    """
    Compute BM25-like scores for a single query across all docs.
    This is a simplified implementation suitable for small corpora.
    """
    if not corpus_tokens:
        return []

    N = len(corpus_tokens)
    qtokens = _tokenize(query)
    if not qtokens:
        return [0.0] * N

    scores = [0.0] * N

    for term in qtokens:
        df_t = df_counts.get(term, 0)
        if df_t == 0:
            continue

        # IDF with plus-one tricks to avoid zero / div by zero
        idf = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1)

        for i, doc_tokens in enumerate(corpus_tokens):
            freq = doc_tokens.count(term)
            if freq == 0:
                continue

            dl = len(doc_tokens) or 1
            numer = freq * (k1 + 1)
            denom = freq + k1 * (1 - b + b * dl / (avgdl or 1.0))
            scores[i] += idf * (numer / denom)

    return scores


def keyword_search(
    collection,
    query: str,
    top_k: int = 5,
    doc_type_filter: str = "all",
) -> List[Dict[str, Any]]:
    """
    Run BM25-style keyword search over all docs (or filtered by doc_type).
    Returns top_k docs with highest scores.

    Returned docs:
        [{
            "id": ...,
            "source_file": ...,
            "doc_type": ...,
            "text": ...,
            "score": float,
        }, ...]
    """
    query = (query or "").strip()
    if not query:
        return []

    df = _build_corpus_from_chroma(collection, doc_type_filter=doc_type_filter)
    if df.empty:
        return []

    corpus_tokens, doc_ids, doc_types, source_files, df_counts, avgdl = _build_inverted_index(df)
    scores = _bm25_scores_for_query(
        query=query,
        corpus_tokens=corpus_tokens,
        df_counts=df_counts,
        avgdl=avgdl,
    )

    scored_rows = []
    for i, score in enumerate(scores):
        if score <= 0:
            continue
        scored_rows.append(
            {
                "id": doc_ids[i],
                "source_file": source_files[i],
                "doc_type": doc_types[i],
                "text": df.iloc[i]["text"],
                "score": float(score),
            }
        )

    # Sort by score descending, keep top_k
    scored_rows.sort(key=lambda d: d["score"], reverse=True)
    return scored_rows[:top_k]
