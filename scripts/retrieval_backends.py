# retrieval_backends.py — vector + keyword search adapters

from __future__ import annotations
from typing import List, Dict, Any, Optional

from keyword_search import keyword_search  # your BM25 module


def run_vector_search(
    collection,
    query: str,
    top_k: int = 10,
    doc_type_filter: str = "all",
) -> List[Dict[str, Any]]:
    """
    Wrapper around Chroma semantic search to produce normalized 0–1 scores.
    Assumes you are using collection.query() with embeddings.
    """
    query = (query or "").strip()
    if not query:
        return []

    where: Optional[Dict[str, Any]] = None
    dt = (doc_type_filter or "all").lower()
    if dt != "all":
        where = {"doc_type": dt}

    # Use Chroma's query API (adjust n_results as needed)
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    if not docs:
        return []

    # Convert distances to similarity & normalize 0–1
    # Assuming smaller distance = closer.
    # similarity = 1 / (1 + distance)
    sims = [1.0 / (1.0 + float(d)) for d in dists]
    max_sim = max(sims) if sims else 1.0
    if max_sim <= 0:
        max_sim = 1.0

    results_out: List[Dict[str, Any]] = []
    for doc_id, text, meta, sim in zip(ids, docs, metas, sims):
        norm_sim = sim / max_sim
        results_out.append(
            {
                "id": str(doc_id),
                "text": text,
                "metadata": meta or {},
                "vector_score": float(norm_sim),
                "keyword_score": None,  # will be filled by BM25 layer
            }
        )

    return results_out


def run_keyword_search(
    collection,
    query: str,
    top_k: int = 10,
    doc_type_filter: str = "all",
) -> List[Dict[str, Any]]:
    """
    Wrapper around your BM25 keyword_search() to unify schema.
    """
    keyword_results = keyword_search(
        collection=collection,
        query=query,
        top_k=top_k,
        doc_type_filter=doc_type_filter,
    )

    out: List[Dict[str, Any]] = []
    for r in keyword_results:
        out.append(
            {
                "id": r["id"],
                "text": r["text"],
                "metadata": {
                    "doc_type": r.get("doc_type"),
                    "source_file": r.get("source_file"),
                },
                "vector_score": None,
                "keyword_score": float(r["score"]),  # already 0–1 normalized
            }
        )
    return out
