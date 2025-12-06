# hybrid_retrieval.py — query-aware hybrid retrieval for DocuMind Lite

from __future__ import annotations
from typing import List, Dict, Any, Optional
from collections import defaultdict

from query_analysis import classify_query
from retrieval_backends import run_vector_search, run_keyword_search


def merge_results(
    vector_results: List[Dict[str, Any]],
    keyword_results: List[Dict[str, Any]],
    alpha: float,
) -> List[Dict[str, Any]]:
    """
    Merge vector + keyword results by document ID.
    alpha = weight for vector_score (0–1).
    (1 - alpha) = weight for keyword_score.
    """
    alpha = max(0.0, min(1.0, alpha))

    by_id: Dict[str, Dict[str, Any]] = {}

    # Index vector results
    for r in vector_results:
        doc_id = r["id"]
        by_id.setdefault(doc_id, {
            "id": doc_id,
            "text": r["text"],
            "metadata": r.get("metadata", {}),
            "vector_score": 0.0,
            "keyword_score": 0.0,
        })
        by_id[doc_id]["vector_score"] = max(
            by_id[doc_id]["vector_score"], r.get("vector_score") or 0.0
        )

    # Index keyword results
    for r in keyword_results:
        doc_id = r["id"]
        by_id.setdefault(doc_id, {
            "id": doc_id,
            "text": r["text"],
            "metadata": r.get("metadata", {}),
            "vector_score": 0.0,
            "keyword_score": 0.0,
        })
        by_id[doc_id]["keyword_score"] = max(
            by_id[doc_id]["keyword_score"], r.get("keyword_score") or 0.0
        )

    merged: List[Dict[str, Any]] = []
    for doc_id, r in by_id.items():
        v = r["vector_score"] or 0.0
        k = r["keyword_score"] or 0.0
        hybrid = alpha * v + (1.0 - alpha) * k
        r["hybrid_score"] = hybrid
        merged.append(r)

    merged.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return merged


def choose_alpha(query_type: str) -> float:
    """
    Decide how much weight to give vector vs keyword based on query type.
    Returns alpha in [0, 1].
    alpha close to 1 → more semantic.
    alpha close to 0 → more keyword.
    """
    if query_type == "exact_lookup":
        return 0.2  # mostly keyword/BM25
    if query_type == "semantic_qa":
        return 0.7  # mostly vector
    if query_type in ("count_query", "filter_stats"):
        return 0.5  # used only for context, counts are handled separately
    return 0.6  # default


def hybrid_retrieve(
    collection,
    query: str,
    top_k: int = 10,
    doc_type_filter: str = "all",
    use_reranker: bool = False,
    reranker_fn: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Main hybrid retrieval function.

    Returns list of docs:
        {
          "id": str,
          "text": str,
          "metadata": dict,
          "vector_score": float,
          "keyword_score": float,
          "hybrid_score": float,
        }
    """
    query = (query or "").strip()
    if not query:
        return []

    qinfo = classify_query(query)
    qtype = qinfo["type"]

    # Run vector + BM25
    vector_results = run_vector_search(
        collection=collection,
        query=query,
        top_k=top_k,
        doc_type_filter=doc_type_filter,
    )

    keyword_results = run_keyword_search(
        collection=collection,
        query=query,
        top_k=top_k,
        doc_type_filter=doc_type_filter,
    )

    # Merge
    alpha = choose_alpha(qtype)
    merged = merge_results(vector_results, keyword_results, alpha=alpha)

    # Optional reranking using cross-encoder / external model
    if use_reranker and reranker_fn is not None and merged:
        # Take top N, rerank by text relevance
        top_for_rerank = merged[: min(top_k, 20)]
        reranked = reranker_fn(query, top_for_rerank)  # you will implement the function later
        return reranked[:top_k]

    return merged[:top_k]
