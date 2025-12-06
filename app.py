# app.py — DocuMind Lite (Advanced Hybrid RAG App + PDF Viewer + Compare)
# Multi-document intelligence over invoices, resumes & contracts.

from __future__ import annotations

import os
import re
import base64
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import chromadb
import pandas as pd
import streamlit as st
from chromadb.utils import embedding_functions
from openai import OpenAI

from ocr_engine import ocr_and_index_upload
from keyword_search import keyword_search
from reranker import embedding_rerank  # uses embeddings for reranking

# ===================== CONFIG =====================

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "index" / "chroma"

# Unified collection for ALL docs (invoice + resume + contract + other)
COLLECTION_NAME = "documents"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Allow more characters per document in the context
MAX_CONTEXT_CHARS_PER_DOC = 3000

DEFAULT_MODEL = "gpt-4o-mini"

STRUCTURED_INVOICES_PATH = BASE_DIR / "data" / "structured" / "invoices.csv"


def get_api_key() -> str:
    """
    Retrieve OpenAI key in a robust way:
    1) First try Streamlit secrets
    2) Then environment variable
    """
    key = ""
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        key = os.getenv("OPENAI_API_KEY", "")
    return key.strip()


OPENAI_API_KEY = get_api_key()
HAS_API_KEY = bool(OPENAI_API_KEY)

# ===================== SANITIZATION / PII HELPERS =====================


def clean_text(text: str) -> str:
    """Basic cleaning for noisy OCR text."""
    if not text:
        return ""
    text = text.replace("\x0c", " ")
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def mask_pii(text: str) -> str:
    """Mask emails and phone numbers in text."""
    if not text:
        return ""

    text = re.sub(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "***@***",
        text,
    )

    text = re.sub(
        r"\b(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b",
        "XXX-XXX-XXXX",
        text,
    )

    return text


# ===================== DATA HELPERS =====================


@st.cache_data(show_spinner=False)
def load_invoices_df() -> Optional[pd.DataFrame]:
    """Load structured invoices CSV if present."""
    if not STRUCTURED_INVOICES_PATH.exists():
        return None
    try:
        df = pd.read_csv(STRUCTURED_INVOICES_PATH)
        return df
    except Exception as e:
        print(f"Error loading invoices.csv: {e}")
        return None


# ===================== CHROMA / INDEX HELPERS =====================


@st.cache_resource(show_spinner=False)
def get_chroma_collection():
    """Return the unified Chroma collection for all documents (cached)."""
    if not INDEX_DIR.exists():
        raise RuntimeError(
            f"Index directory not found: {INDEX_DIR}\n"
            "Tip: run your indexing script (e.g. scripts/build_index_all.py)."
        )

    client = chromadb.PersistentClient(path=str(INDEX_DIR))

    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=emb_fn,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load Chroma collection '{COLLECTION_NAME}': {e}")

    return collection


def retrieve_single_doc(collection, doc_id: str) -> List[dict]:
    """Retrieve a single document by id (for 'chat with specific document')."""
    try:
        results = collection.get(
            ids=[doc_id],
            include=["documents", "metadatas"],
        )
    except Exception:
        return []

    docs_raw = results.get("documents", [])
    metas = results.get("metadatas", [])
    ids = results.get("ids", [])

    docs: List[dict] = []
    for doc_id, meta, text in zip(ids, metas, docs_raw):
        docs.append(
            {
                "id": str(doc_id),
                "source_file": str(meta.get("source_file", "unknown")),
                "doc_type": str(meta.get("doc_type", "other")),
                "text": text or "",
            }
        )
    return docs


def get_metadata_df(collection, doc_type_filter: str = "all") -> pd.DataFrame:
    """
    Fetch metadata (no text, no embeddings) for listing/searching documents.

    Returns DataFrame with columns:
      - doc_id
      - doc_type
      - source_file
      - rel_path
      - text_len
    """
    get_kwargs: Dict[str, Any] = {
        "include": ["metadatas"],  # 'ids' is not allowed here; it's always returned
    }

    dt = (doc_type_filter or "all").lower()
    if dt != "all":
        get_kwargs["where"] = {"doc_type": dt}

    results = collection.get(**get_kwargs)
    ids = results.get("ids", []) or []
    metas = results.get("metadatas", []) or []

    rows = []
    for doc_id, meta in zip(ids, metas):
        rows.append(
            {
                "doc_id": str(doc_id),
                "doc_type": str(meta.get("doc_type", "other")),
                "source_file": str(meta.get("source_file", "")),
                "rel_path": str(meta.get("rel_path", "")),
                "text_len": int(meta.get("text_len", 0)),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["doc_id", "doc_type", "source_file", "rel_path", "text_len"]
        )

    return pd.DataFrame(rows)


# ===================== ADVANCED RAG (PHASE 3 — RAG 3.0) =====================

# ---- 1. Query type classification ----

def classify_query_type(query: str) -> str:
    """
    Rough query type classifier:
    Returns one of:
      - "exact_lookup"
      - "semantic_qa"
      - "count_query"
      - "filter_stats"
      - "unknown"
    """
    q = (query or "").strip().lower()
    if not q:
        return "unknown"

    # Count questions
    if any(p in q for p in ["how many", "number of", "count of"]):
        return "count_query"

    # Aggregation / stats
    if any(p in q for p in ["total amount", "sum of", "total value", "maximum", "minimum", "average", "avg"]):
        return "filter_stats"

    # ID-style / exact lookup (short alphanumeric)
    import re as _re
    id_like = bool(_re.search(r"[a-z]{2,}\-\d{2,}", q))
    has_letters_and_digits = bool(_re.search(r"[a-z]", q) and _re.search(r"\d", q))
    short_query = len(q) <= 30
    if (id_like or has_letters_and_digits) and short_query:
        return "exact_lookup"

    # Question-style
    if any(q.startswith(w) for w in ["what", "who", "when", "where", "why", "how"]):
        return "semantic_qa"

    # Long natural text → semantic
    if len(q.split()) >= 6:
        return "semantic_qa"

    return "unknown"


# ---- 2. Vector search wrapper (Chroma) ----

def _run_vector_search(
    collection,
    question: str,
    top_k: int,
    doc_type_filter: str = "all",
) -> List[Dict[str, Any]]:
    """
    Wrapper around Chroma semantic search to produce normalized 0–1 scores.
    """
    question = (question or "").strip()
    if not question:
        return []

    query_kwargs: Dict[str, Any] = {
        "query_texts": [question],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }

    dt = (doc_type_filter or "all").lower()
    if dt != "all":
        query_kwargs["where"] = {"doc_type": dt}

    try:
        results = collection.query(**query_kwargs)
    except Exception:
        return []

    docs_raw = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]
    ids = (results.get("ids") or [[]])[0]

    if not docs_raw:
        return []

    # Convert distances to similarity and normalize
    sims: List[float] = []
    for d in dists:
        try:
            val = float(d)
        except Exception:
            val = 10.0
        sims.append(1.0 / (1.0 + val))

    max_sim = max(sims) if sims else 1.0
    if max_sim <= 0:
        max_sim = 1.0

    out: List[Dict[str, Any]] = []
    for doc_id, text, meta, sim in zip(ids, docs_raw, metas, sims):
        norm_sim = sim / max_sim
        out.append(
            {
                "id": str(doc_id),
                "text": text or "",
                "metadata": meta or {},
                "vector_score": float(norm_sim),
                "keyword_score": 0.0,
            }
        )
    return out


# ---- 3. Keyword (BM25) search wrapper ----

def _run_keyword_search(
    collection,
    question: str,
    top_k: int,
    doc_type_filter: str = "all",
) -> List[Dict[str, Any]]:
    """
    Wrapper around your BM25 keyword_search() to unify schema with vector search.
    """
    try:
        kw_results = keyword_search(
            collection=collection,
            query=question,
            top_k=top_k,
            doc_type_filter=doc_type_filter,
        )
    except Exception:
        return []

    if not kw_results:
        return []

    out: List[Dict[str, Any]] = []
    for r in kw_results:
        meta = {
            "doc_type": r.get("doc_type", "other"),
            "source_file": r.get("source_file", ""),
        }
        out.append(
            {
                "id": str(r["id"]),
                "text": r.get("text", "") or "",
                "metadata": meta,
                "vector_score": 0.0,
                "keyword_score": float(r.get("score", 0.0)),  # already normalized 0–1
            }
        )
    return out


# ---- 4. Merge vector + BM25 into hybrid result ----

def _choose_alpha(query_type: str) -> float:
    """
    Decide weight for vector vs keyword based on query type.
    alpha close to 1 → more vector/semantic.
    alpha close to 0 → more keyword/exact.
    """
    if query_type == "exact_lookup":
        return 0.25
    if query_type == "semantic_qa":
        return 0.7
    if query_type in ("count_query", "filter_stats"):
        return 0.5
    return 0.6  # default


def hybrid_retrieve(
    collection,
    question: str,
    top_k: int = 3,
    doc_type_filter: str = "all",
    use_reranker: bool = False,
) -> List[dict]:
    """
    Query-aware hybrid retrieval:
    - Uses both vector & BM25
    - Weights them based on query type
    - Optional embedding-based reranker
    - Returns docs with hybrid_score plus doc metadata
    """
    question = (question or "").strip()
    if not question:
        return []

    qtype = classify_query_type(question)

    vector_results = _run_vector_search(
        collection=collection,
        question=question,
        top_k=top_k,
        doc_type_filter=doc_type_filter,
    )

    keyword_results = _run_keyword_search(
        collection=collection,
        question=question,
        top_k=top_k,
        doc_type_filter=doc_type_filter,
    )

    alpha = _choose_alpha(qtype)

    # Merge by id
    by_id: Dict[str, Dict[str, Any]] = {}

    # Seed with vector results
    for r in vector_results:
        doc_id = r["id"]
        meta = r.get("metadata", {}) or {}
        by_id[doc_id] = {
            "id": doc_id,
            "text": r.get("text", "") or "",
            "doc_type": str(meta.get("doc_type", "other")),
            "source_file": str(meta.get("source_file", "")),
            "vector_score": float(r.get("vector_score", 0.0)),
            "keyword_score": 0.0,
        }

    # Add keyword results
    for r in keyword_results:
        doc_id = r["id"]
        if doc_id not in by_id:
            meta = r.get("metadata", {}) or {}
            by_id[doc_id] = {
                "id": doc_id,
                "text": r.get("text", "") or "",
                "doc_type": str(meta.get("doc_type", "other")),
                "source_file": str(meta.get("source_file", "")),
                "vector_score": 0.0,
                "keyword_score": float(r.get("keyword_score", 0.0)),
            }
        else:
            by_id[doc_id]["keyword_score"] = float(r.get("keyword_score", 0.0))

    # Compute hybrid score and sort
    docs: List[dict] = []
    for doc_id, info in by_id.items():
        v = info.get("vector_score", 0.0) or 0.0
        k = info.get("keyword_score", 0.0) or 0.0
        hybrid_score = alpha * v + (1.0 - alpha) * k
        info["hybrid_score"] = float(hybrid_score)
        docs.append(info)

    docs.sort(key=lambda d: d.get("hybrid_score", 0.0), reverse=True)

    # 🔥 Optional embedding-based rerank on top of hybrid_score
    if use_reranker and HAS_API_KEY and docs:
        docs = embedding_rerank(
            query=question,
            docs=docs[: min(top_k, 10)],   # rerank only top-N
            api_key=OPENAI_API_KEY,
        )

    # Limit to top_k
    return docs[:top_k]


# ===================== CONTEXT BUILDER + LLM CALL =====================


def build_context(docs: List[dict]) -> str:
    """
    Build a safe, cleaned, PII-masked context string from retrieved docs.

    Each document is clearly separated and numbered so the model
    can reference them without confusion, with labels [DOC1], [DOC2], ...
    """
    if not docs:
        return "No relevant documents retrieved."

    blocks: List[str] = []
    for i, d in enumerate(docs, start=1):
        label = f"DOC{i}"
        raw = d.get("text", "") or ""
        cleaned = clean_text(raw)
        safe = mask_pii(cleaned)
        snippet = safe[:MAX_CONTEXT_CHARS_PER_DOC]

        blocks.append(
            f"=== DOCUMENT {i} [{label}] ===\n"
            f"ID: {d.get('id', '')}\n"
            f"Type: {d.get('doc_type', 'other')}\n"
            f"File: {d.get('source_file', 'unknown')}\n\n"
            f"TEXT:\n{snippet}"
        )

    return "\n\n".join(blocks)


def call_llm(
    question: str,
    context: str,
    model: str = DEFAULT_MODEL,
    strict: bool = True,
) -> Tuple[bool, str]:
    """
    Call OpenAI chat completion using cleaned + masked context.

    strict=True  -> model is NOT allowed to guess beyond context.
    Returns (success, message).
    """
    if not HAS_API_KEY:
        return False, (
            "OPENAI_API_KEY is not set.\n\n"
            "Set it as:\n"
            "• Local PowerShell:  $env:OPENAI_API_KEY = 'sk-...'\n"
            "• Streamlit Cloud:   Settings → Secrets → OPENAI_API_KEY='sk-...'"
        )

    client = OpenAI(api_key=OPENAI_API_KEY)

    base_rules = [
        "You are DocuMind Lite, an assistant that answers questions about documents.",
        "Use ONLY the information in the provided documents context.",
        "Never invent invoice numbers, totals, dates, or names.",
        "If something is not explicitly present in the context, you MUST say you do not know.",
        "If you are not sure, answer with: 'The documents provided do not contain this information clearly.'",
        "If multiple documents are relevant, clearly say which document you are talking about.",
        "The context labels documents as [DOC1], [DOC2], etc. Always cite these labels when using information.",
    ]

    if strict:
        base_rules.append(
            "You are in STRICT MODE: you are forbidden from hallucinating or guessing."
        )
    else:
        base_rules.append(
            "You may infer a little, but still clearly mention when you are not fully sure."
        )

    system_prompt = (
        "\n".join(base_rules)
        + "\n\n"
        "- If the document is an invoice, include invoice number, date, and total when possible.\n"
        "- If the document is a resume, focus on skills, experience, and education.\n"
        "- If the document is a contract, focus on parties, dates, and obligations.\n"
        "- When you state a fact, include the matching label in square brackets at the end, e.g. [DOC1]."
    )

    user_prompt = f"""
DOCUMENT CONTEXT START
{context}
DOCUMENT CONTEXT END

User question:
{question}

Answering rules:
- Base your answer ONLY on the context between CONTEXT START and CONTEXT END.
- Quote numbers, dates, and names exactly as shown.
- Whenever you use a fact from a document, add the label at the end like [DOC1], [DOC2], etc.
- If the answer is missing or ambiguous, explicitly say you cannot answer from these documents.
- Keep the answer in 2–8 short sentences.
"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as e:
        return False, f"OpenAI API Error: {e}"

    if not resp or not resp.choices:
        return False, "No valid response from the AI."

    return True, (resp.choices[0].message.content or "(No answer returned.)")


# ---------- Special: count questions like "how many invoices" ----------


def handle_count_question(question: str, collection) -> Optional[str]:
    """
    Detect questions like:
      - 'how many resumes'
      - 'total invoices'
      - 'number of contracts'

    DO NOT trigger on:
      - 'total amount'
      - 'invoice total'
      - 'amount due'
      - 'grand total'
    """
    q = question.lower()

    # SAFE LIST of phrases that truly mean counting documents
    count_phrases = [
        "how many invoices",
        "how many invoice",
        "how many resumes",
        "how many resume",
        "how many contracts",
        "how many contract",
        "number of invoices",
        "number of invoice",
        "number of resumes",
        "number of resume",
        "number of contracts",
        "number of contract",
        "count invoices",
        "count invoice",
        "count resumes",
        "count resume",
        "count contracts",
        "count contract",
        "total invoices",
        "total resumes",
        "total contracts",
    ]

    if not any(phrase in q for phrase in count_phrases):
        return None

    doc_type = None
    if "resume" in q or "resumes" in q:
        doc_type = "resume"
    elif "contract" in q or "contracts" in q:
        doc_type = "contract"
    elif "invoice" in q or "invoices" in q:
        doc_type = "invoice"

    if not doc_type:
        return None

    try:
        results = collection.get(
            where={"doc_type": doc_type},
            include=["metadatas"],  # ids are always returned implicitly
        )
        num_docs = len(results.get("ids", []))
    except Exception:
        return "I tried to count documents, but there was an error reading the index."

    return f"There are **{num_docs} {doc_type} document(s)** currently indexed in DocuMind Lite."


# ===================== STREAMLIT LAYOUT HELPERS =====================


def sidebar_layout(invoices_df: Optional[pd.DataFrame]) -> Tuple[str, int]:
    """Build the sidebar and return (doc_type_for_qa, top_k)."""
    with st.sidebar:
        st.image(
            "https://em-content.zobj.net/source/microsoft-teams/363/page-facing-up_1f4c4.png",
            width=48,
        )

        st.title("DocuMind Lite ⚡")
        st.caption("Smart document Q&A for invoices, resumes & contracts.")

        st.markdown("### ⚙️ Q&A Settings")

        doc_type_for_qa = st.selectbox(
            "Limit Q&A + document search to type",
            ["all", "invoice", "resume", "contract"],
            index=0,
        )

        top_k = st.slider("Top-k documents to retrieve for Q&A", 1, 10, 3)

        st.markdown("---")
        st.markdown("### 📤 Upload → OCR → Index")

        uploaded = st.file_uploader(
            "Upload invoice/resume/contract (PDF/PNG/JPG)",
            type=["pdf", "png", "jpg", "jpeg"],
            help="Uploaded file will be OCR'd and added to the vector index.",
        )
        upload_doc_type = st.selectbox(
            "Label uploaded document as",
            ["auto-detect", "invoice", "resume", "contract", "other"],
            index=0,
            help="Use 'auto-detect' to let the app guess based on text.",
        )

        if st.button("Run OCR + index uploaded file", key="process_upload"):
            if uploaded is None:
                st.warning("Please upload a file first.")
            else:
                try:
                    collection = get_chroma_collection()
                except Exception as e:
                    st.error(f"Could not load index: {e}")
                else:
                    ok, msg = ocr_and_index_upload(
                        uploaded, upload_doc_type, collection
                    )
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

        st.markdown("---")
        st.markdown("### 📊 Structured invoices (CSV)")
        if invoices_df is not None and not invoices_df.empty:
            st.write(f"Total invoices in CSV: **{len(invoices_df)}**")
            csv_bytes = invoices_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download invoices.csv",
                data=csv_bytes,
                file_name="invoices.csv",
                mime="text/csv",
            )
        else:
            st.info("`invoices.csv` not found or empty. Upload some invoices to generate it.")

        st.markdown("---")
        if HAS_API_KEY:
            st.success("OPENAI_API_KEY found — LLM Q&A is **enabled** ✅")
        else:
            st.error(
                "OPENAI_API_KEY is **missing** ❌\n\n"
                "Q&A fallback: index-based counts & search still work.\n"
                "Add your key in `.env` or Streamlit `secrets.toml`."
            )

        st.markdown("---")
        st.markdown("### 👤 About the author")

        # Stylish GitHub + LinkedIn badges
        st.markdown(
            """
<b>Aditya Jadhav</b><br/>
<div style="display: flex; gap: 8px; margin-top: 6px;">
  <a href="https://github.com/AdityaJadhav-ds" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white"
         height="24">
  </a>
  <a href="https://www.linkedin.com/in/aditya-jadhav-6775702b4" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?logo=linkedin&logoColor=white"
         height="24">
  </a>
</div>
""",
            unsafe_allow_html=True,
        )

    return doc_type_for_qa, top_k


def render_kpi_cards(meta_df: pd.DataFrame) -> None:
    """Small KPI cards with counts per document type."""
    if meta_df.empty:
        return

    total_docs = len(meta_df)
    by_type = meta_df["doc_type"].value_counts().to_dict()

    invoice_ct = by_type.get("invoice", 0)
    resume_ct = by_type.get("resume", 0)
    contract_ct = by_type.get("contract", 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Total docs", total_docs)
    c2.metric("🧾 Invoices", invoice_ct)
    c3.metric("👤 Resumes", resume_ct)
    c4.metric("📑 Contracts", contract_ct)


# ===================== PDF / IMAGE VIEWER HELPERS =====================


def _resolve_doc_path(meta: Dict[str, Any]) -> Optional[Path]:
    """
    Try to resolve the original file path on disk using metadata.
    1) Try common locations based on rel_path and source_file.
    2) If not found, search the entire project tree for the filename.
    """
    candidates: List[Path] = []

    rel = str(meta.get("rel_path", "") or "").strip()
    src = str(meta.get("source_file", "") or "").strip()

    if rel:
        candidates.append(BASE_DIR / rel)
        candidates.append(BASE_DIR / "data" / rel)

    if src:
        candidates.append(BASE_DIR / src)
        candidates.append(BASE_DIR / "data" / src)
        candidates.append(BASE_DIR / "data" / "raw" / src)
        candidates.append(BASE_DIR / "uploads" / src)
        candidates.append(BASE_DIR / "docs" / src)

    # First pass: direct candidates
    seen: set[Path] = set()
    for p in candidates:
        p = Path(p)
        if p in seen:
            continue
        seen.add(p)
        if p.exists():
            return p

    # Second pass: search by filename anywhere under project
    # Use the most informative name we have (src or rel)
    name_candidate = (src or rel).strip()
    name_candidate = Path(name_candidate).name if name_candidate else ""
    if name_candidate:
        try:
            for root in [BASE_DIR, BASE_DIR / "data", BASE_DIR / "uploads"]:
                root = Path(root)
                if not root.exists():
                    continue
                for found in root.rglob(name_candidate):
                    if found.is_file():
                        return found
        except Exception:
            pass

    return None


def _show_pdf(path: Path) -> None:
    """Display a PDF inline using base64 iframe + give download."""
    with open(path, "rb") as f:
        pdf_bytes = f.read()
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_display = f"""
    <iframe src="data:application/pdf;base64,{b64}" width="100%" height="800px" type="application/pdf"></iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)
    st.download_button(
        "⬇️ Download file",
        data=pdf_bytes,
        file_name=path.name,
        mime="application/pdf",
    )


def render_viewer_tab() -> None:
    """PDF / image viewer tab."""
    st.subheader("📄 PDF / Image viewer")

    try:
        collection = get_chroma_collection()
    except Exception as e:
        st.error(f"Could not load collection: {e}")
        return

    meta_df = get_metadata_df(collection, doc_type_filter="all")

    if meta_df.empty:
        st.info("No documents in index yet.")
        return

    meta_df = meta_df.copy()
    meta_df["display"] = meta_df.apply(
        lambda r: f"{r['doc_id']} — {Path(str(r['source_file'])).name}", axis=1
    )

    selected_label = st.selectbox(
        "Select a document to view",
        meta_df["display"].tolist(),
        key="viewer_select_doc",
    )

    row = meta_df[meta_df["display"] == selected_label].iloc[0].to_dict()
    path = _resolve_doc_path(row)

    st.markdown(
        f"**Doc ID:** `{row.get('doc_id', '')}`  |  "
        f"Type: `{row.get('doc_type', '')}`  |  "
        f"Source file meta: `{row.get('source_file', '')}`"
    )

    if path is None:
        st.warning(
            "Original PDF/image file not found on disk.\n\n"
            "Fix tips:\n"
            "1) Make sure the actual file is inside this project folder.\n"
            "2) If needed, move it into a folder like 'data', 'data/raw', or 'uploads'.\n"
            "3) Rebuild the index later to store better rel_path metadata."
        )
        return

    st.write(f"**Resolved path:** `{path}`")

    suffix = path.suffix.lower()
    if suffix in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
        st.image(str(path), caption=path.name, use_container_width=True)
        with open(path, "rb") as f:
            img_bytes = f.read()
        st.download_button(
            "⬇️ Download file",
            data=img_bytes,
            file_name=path.name,
        )
    elif suffix == ".pdf":
        _show_pdf(path)
    else:
        st.info("This file is not a PDF or image. You can still download it.")
        with open(path, "rb") as f:
            data = f.read()
        st.download_button(
            "⬇️ Download file",
            data=data,
            file_name=path.name,
        )


# ===================== TABS =====================


def render_qa_tab(doc_type_for_qa: str, top_k: int) -> None:
    """Main RAG Q&A tab (now using advanced hybrid retrieval + citations + optional reranker)."""
    st.subheader("💬 Ask a question about your documents")

    strict_mode = st.toggle(
        "Strict mode (only answer from documents, no guessing)",
        value=True,
        help="Turn this OFF only if you want the model to infer slightly when context is weak.",
    )

    use_reranker = st.checkbox(
        "Use advanced reranker (better ranking, slightly more API cost)",
        value=False,
        help="Re-orders top documents using OpenAI embeddings for higher answer quality.",
    )

    try:
        collection = get_chroma_collection()
    except Exception as e:
        st.error(f"Error loading Chroma collection: {e}")
        return

    # Optional: chat with a specific document
    specific_mode = st.checkbox(
        "Chat with a specific document", value=False, help="Scope Q&A to one doc id."
    )
    specific_doc_id: Optional[str] = None

    if specific_mode:
        meta_df_specific = get_metadata_df(collection, doc_type_filter=doc_type_for_qa)
        if meta_df_specific.empty:
            st.info(
                f"No documents available for type '{doc_type_for_qa}' to select from."
            )
            specific_mode = False
        else:
            options = []
            id_map: Dict[str, str] = {}
            for _, row in meta_df_specific.iterrows():
                label = f"{row['doc_id']} — {Path(row['source_file']).name}"
                options.append(label)
                id_map[label] = row["doc_id"]

            selected_label = st.selectbox(
                "Select a document", options, index=0, key="qa_specific_doc"
            )
            specific_doc_id = id_map[selected_label]

    sample_qs = [
        "What is the total amount on the latest invoice?",
        "What are the main skills in the resume?",
        "Who are the parties and effective date in the contract?",
        "How many invoices are indexed?",
    ]

    col_q1, col_q2 = st.columns([3, 1])

    with col_q1:
        question = st.text_input(
            "Question",
            placeholder="Type your question, or pick a sample from the right…",
            key="qa_question",
        )

    with col_q2:
        chosen_sample = st.selectbox(
            "Sample question (optional)",
            ["(none)"] + sample_qs,
            index=0,
        )
        if chosen_sample != "(none)" and not question:
            st.session_state["qa_question"] = chosen_sample
            question = chosen_sample

    ask_clicked = st.button("🚀 Ask")

    if "qa_history" not in st.session_state:
        st.session_state["qa_history"] = []

    if ask_clicked and question.strip():
        # 1) Index-based count questions (no LLM), only when not scoping to one doc
        if not specific_mode:
            special_answer = handle_count_question(question, collection)
            if special_answer is not None:
                st.markdown("### ✅ Answer (from index metadata)")
                st.write(special_answer)
                st.session_state.qa_history.insert(
                    0, {"question": question, "answer": special_answer}
                )
                return

        # 2) Hybrid RAG flow
        with st.spinner("Retrieving relevant documents from index…"):
            if specific_mode and specific_doc_id:
                docs = retrieve_single_doc(collection, specific_doc_id)
            else:
                docs = hybrid_retrieve(
                    collection=collection,
                    question=question,
                    top_k=top_k,
                    doc_type_filter=doc_type_for_qa,
                    use_reranker=use_reranker,
                )

        if not docs:
            st.warning(
                "No relevant documents found for this question.\n"
                "Tip: make sure OCR was run and `scripts/build_index_all.py` was executed."
            )
            return

        context = build_context(docs)

        st.markdown("---")
        success, answer = call_llm(question, context, strict=strict_mode)

        if success:
            st.markdown("### ✅ Answer")
            st.write(answer)
        else:
            st.markdown("### ❌ Could not get an answer")
            st.error(answer)

        # Save to history
        st.session_state.qa_history.insert(
            0, {"question": question, "answer": answer}
        )

        with st.expander("📎 Show context documents (masked text)"):
            for i, d in enumerate(docs, start=1):
                raw = d.get("text", "") or ""
                cleaned = clean_text(raw)
                safe = mask_pii(cleaned)
                snippet = safe[:700]
                label = f"DOC{i}"
                st.markdown(
                    f"**[{label}] {d.get('doc_type', 'OTHER').upper()} — {d.get('id', '')} — {d.get('source_file', '')}**"
                )
                st.text(snippet + ("..." if len(safe) > 700 else ""))

        # ---------- NEW: Retrieval scores debug ----------
        with st.expander("🔍 Retrieval scores (debug)"):
            for i, d in enumerate(docs, start=1):
                label = f"DOC{i}"
                st.markdown(f"**{label} — ID: {d.get('id', '')}**")
                st.write(
                    {
                        "doc_type": d.get("doc_type"),
                        "source_file": d.get("source_file"),
                        "vector_score": d.get("vector_score"),
                        "keyword_score": d.get("keyword_score"),
                        "hybrid_score": d.get("hybrid_score"),
                        "rerank_score": d.get("rerank_score", None),
                    }
                )

        # ---------- Keyword/BM25 search debug view ----------
        with st.expander("🧪 Keyword search (BM25-style, debug)"):
            try:
                kw_docs = keyword_search(
                    collection,
                    query=question,
                    top_k=top_k,
                    doc_type_filter=doc_type_for_qa,
                )
                if not kw_docs:
                    st.write("No keyword matches found.")
                else:
                    for d in kw_docs:
                        st.markdown(
                            f"- **[{d['doc_type'].upper()}] {d['id']}** "
                            f"({d['source_file']}) — score: `{d['score']:.3f}`"
                        )
            except Exception as e:
                st.error(f"Keyword search error: {e}")

    st.markdown("---")

    if st.session_state.qa_history:
        st.markdown("#### 🕒 Recent Q&A (local only)")
        for item in st.session_state.qa_history[:5]:
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {item['answer']}")
            st.markdown("---")


def render_compare_tab() -> None:
    """Document comparison tab — pick Doc A & Doc B and see differences."""
    st.subheader("📑 Compare two documents (A vs B)")

    try:
        collection = get_chroma_collection()
    except Exception as e:
        st.error(f"Could not load collection: {e}")
        return

    meta_df = get_metadata_df(collection, doc_type_filter="all")

    if meta_df.empty:
        st.info("No documents in index yet.")
        return

    meta_df = meta_df.copy()
    meta_df["display"] = meta_df.apply(
        lambda r: f"{r['doc_type']} | {r['doc_id']} — {Path(str(r['source_file'])).name}",
        axis=1,
    )

    col_a, col_b = st.columns(2)

    with col_a:
        doc_a_label = st.selectbox(
            "Select Document A",
            meta_df["display"].tolist(),
            key="compare_doc_a",
        )

    with col_b:
        doc_b_label = st.selectbox(
            "Select Document B",
            meta_df["display"].tolist(),
            key="compare_doc_b",
        )

    if doc_a_label == doc_b_label:
        st.warning("Select two different documents to compare.")
        return

    row_a = meta_df[meta_df["display"] == doc_a_label].iloc[0]
    row_b = meta_df[meta_df["display"] == doc_b_label].iloc[0]

    docs_a = retrieve_single_doc(collection, row_a["doc_id"])
    docs_b = retrieve_single_doc(collection, row_b["doc_id"])

    if not docs_a or not docs_b:
        st.error("Could not retrieve one or both documents from index.")
        return

    doc_a = docs_a[0]
    doc_b = docs_b[0]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📄 Document A")
        st.markdown(
            f"**ID:** `{doc_a['id']}`  |  "
            f"Type: `{doc_a['doc_type']}`  |  "
            f"File: `{doc_a['source_file']}`"
        )
        a_text = mask_pii(clean_text(doc_a.get("text", "") or ""))
        st.text(a_text[:1200] + ("..." if len(a_text) > 1200 else ""))

    with col2:
        st.markdown("### 📄 Document B")
        st.markdown(
            f"**ID:** `{doc_b['id']}`  |  "
            f"Type: `{doc_b['doc_type']}`  |  "
            f"File: `{doc_b['source_file']}`"
        )
        b_text = mask_pii(clean_text(doc_b.get("text", "") or ""))
        st.text(b_text[:1200] + ("..." if len(b_text) > 1200 else ""))

    st.markdown("---")

    compare_clicked = st.button("🔍 Compare A vs B (LLM)")

    if compare_clicked:
        if not HAS_API_KEY:
            st.error("OPENAI_API_KEY is missing — cannot run LLM comparison.")
            return

        # Build context with DOC1 = A, DOC2 = B
        docs_for_ctx = [
            {
                "id": doc_a["id"],
                "doc_type": doc_a["doc_type"],
                "source_file": doc_a["source_file"],
                "text": doc_a["text"],
            },
            {
                "id": doc_b["id"],
                "doc_type": doc_b["doc_type"],
                "source_file": doc_b["source_file"],
                "text": doc_b["text"],
            },
        ]
        ctx = build_context(docs_for_ctx)

        question = (
            "Compare DOCUMENT 1 [DOC1] and DOCUMENT 2 [DOC2]. "
            "Explain the main differences in: parties/names, dates, amounts, "
            "key clauses or sections, and any important changes in responsibilities "
            "or terms. If they are resumes, compare roles, skills, and experience. "
            "Use bullet points and always cite [DOC1] or [DOC2] for each point."
        )

        with st.spinner("Calling LLM to compare documents…"):
            ok, answer = call_llm(
                question=question,
                context=ctx,
                strict=False,  # allow some reasoning but still grounded
            )

        if ok:
            st.markdown("### ✅ Comparison result")
            st.write(answer)
        else:
            st.markdown("### ❌ Could not compare documents")
            st.error(answer)


def render_browse_tab(doc_type_for_qa: str) -> None:
    """Browse & search tab backed by Chroma metadata."""
    st.subheader("🔍 Browse & search documents")

    try:
        collection_for_table = get_chroma_collection()
    except Exception as e:
        st.info(f"Document search unavailable: {e}")
        return

    meta_df = get_metadata_df(collection_for_table, doc_type_filter=doc_type_for_qa)

    render_kpi_cards(meta_df)

    search_col1, search_col2 = st.columns([2, 1])

    with search_col1:
        search_text = st.text_input(
            "Search by doc_id / filename / path",
            placeholder="e.g., Aaron, resume, contract_01, 2012, 35876",
            key="search_docs_text",
        )

    with search_col2:
        max_rows = st.number_input(
            "Max rows to display",
            min_value=10,
            max_value=1000,
            value=200,
            step=10,
            key="max_rows_docs",
        )

    if not meta_df.empty and search_text.strip():
        s = search_text.lower()
        mask = (
            meta_df["doc_id"].str.lower().str.contains(s)
            | meta_df["source_file"].str.lower().str.contains(s)
            | meta_df["rel_path"].str.lower().str.contains(s)
        )
        meta_df = meta_df[mask]

    meta_df = meta_df.sort_values(["doc_type", "source_file"]).head(int(max_rows))

    if meta_df.empty:
        st.info(
            f"No documents found for type '{doc_type_for_qa}'. "
            "Check that OCR text files exist and the index is built."
        )
    else:
        st.write(
            f"Showing **{len(meta_df)}** document(s) "
            f"for type **{doc_type_for_qa}**."
        )
        st.dataframe(meta_df, use_container_width=True)


def render_invoices_tab(invoices_df: Optional[pd.DataFrame]) -> None:
    """Structured invoice search & basic analytics."""
    st.subheader("📊 Structured invoice search (only invoices)")

    if invoices_df is None or invoices_df.empty:
        st.info(
            "Structured invoice search unavailable: `invoices.csv` not found or empty."
        )
        return

    col1, col2 = st.columns(2)

    with col1:
        inv_search_text = st.text_input(
            "Search invoices by doc_id / file name / invoice number",
            placeholder="e.g., Aaron, 2012, 35876, .pdf",
            key="search_invoices_text",
        )

    with col2:
        min_total = st.number_input(
            "Minimum total amount (optional)",
            min_value=0.0,
            value=0.0,
            step=50.0,
            key="min_total_invoices",
        )

    # Optional date range filter if 'invoice_date' column exists
    date_range = None
    if "invoice_date" in invoices_df.columns:
        with st.expander("📅 Filter by invoice date (optional)"):
            date_range = st.date_input(
                "Invoice date range",
                value=(),
                key="invoice_date_range",
            )

    filtered = invoices_df.copy()

    if inv_search_text.strip():
        s = inv_search_text.lower()
        mask = pd.Series(False, index=filtered.index)

        for col in ["doc_id", "source_file", "invoice_number"]:
            if col in filtered.columns:
                mask = mask | filtered[col].astype(str).str.lower().str.contains(s)

        filtered = filtered[mask]

    if "total_amount" in filtered.columns and min_total > 0:
        try:
            filtered["total_amount_num"] = (
                filtered["total_amount"]
                .astype(str)
                .str.replace(",", "", regex=False)
                .astype(float)
            )
        except Exception:
            filtered["total_amount_num"] = 0.0

        filtered = filtered[filtered["total_amount_num"] >= min_total]

    # Apply date range filter
    if date_range and len(date_range) == 2 and "invoice_date" in filtered.columns:
        try:
            tmp_dates = pd.to_datetime(filtered["invoice_date"], errors="coerce")
            start, end = date_range
            mask = (tmp_dates >= pd.to_datetime(start)) & (
                tmp_dates <= pd.to_datetime(end)
            )
            filtered = filtered[mask]
        except Exception:
            pass

    st.write(f"Found **{len(filtered)}** matching invoices.")
    if "total_amount_num" in filtered.columns:
        filtered = filtered.drop(columns=["total_amount_num"])

    st.dataframe(filtered, use_container_width=True)

    # Simple analytics: total value of filtered invoices
    if "total_amount" in filtered.columns:
        try:
            tmp = (
                filtered["total_amount"]
                .astype(str)
                .str.replace(",", "", regex=False)
                .astype(float)
            )
            total_value = float(tmp.sum())
            st.metric("💰 Sum of totals (filtered)", f"{total_value:,.2f}")
        except Exception:
            pass


def render_admin_tab() -> None:
    """Admin / analytics: overall stats + delete documents."""
    st.subheader("📈 Admin / Analytics")

    try:
        collection = get_chroma_collection()
    except Exception as e:
        st.error(f"Could not load collection: {e}")
        return

    meta_df = get_metadata_df(collection, doc_type_filter="all")

    if meta_df.empty:
        st.info("No documents in index yet.")
        return

    render_kpi_cards(meta_df)

    st.markdown("#### 📊 Documents by type")
    type_counts = meta_df["doc_type"].value_counts()
    st.bar_chart(type_counts)

    st.markdown("#### 🗂️ Latest documents")
    latest = meta_df.sort_values("doc_id", ascending=False).head(20)
    st.dataframe(latest, use_container_width=True)

    st.markdown("#### 🧨 Delete documents from index")
    options = []
    id_map: Dict[str, str] = {}
    for _, row in meta_df.iterrows():
        label = f"{row['doc_id']} — {Path(row['source_file']).name}"
        options.append(label)
        id_map[label] = row["doc_id"]

    selected = st.multiselect(
        "Select document(s) to delete from Chroma index",
        options,
        key="admin_delete_docs",
    )

    if st.button("Delete selected documents", key="btn_delete_docs"):
        if not selected:
            st.warning("No documents selected.")
        else:
            ids_to_delete = [id_map[label] for label in selected]
            try:
                collection.delete(ids=ids_to_delete)
                st.success(f"Deleted {len(ids_to_delete)} document(s) from index.")
            except Exception as e:
                st.error(f"Failed to delete documents: {e}")


# ===================== MAIN =====================


def main() -> None:
    st.set_page_config(
        page_title="DocuMind Lite",
        page_icon="📄",
        layout="wide",
    )

    st.title("📄 DocuMind Lite — Multi-Document Intelligence")
    st.caption(
        "Upload → OCR → Index → Hybrid RAG over invoices, resumes & contracts.\n"
        "Beginner-friendly, but designed like a real-world document AI tool."
    )

    invoices_df = load_invoices_df()
    doc_type_for_qa, top_k = sidebar_layout(invoices_df)

    tab_qa, tab_viewer, tab_compare, tab_browse, tab_invoices, tab_admin = st.tabs(
        [
            "💬 Q&A",
            "📄 Viewer",
            "📑 Compare docs",
            "🔍 Browse documents",
            "📊 Invoices (CSV)",
            "🛠 Admin / Analytics",
        ]
    )

    with tab_qa:
        render_qa_tab(doc_type_for_qa, top_k)

    with tab_viewer:
        render_viewer_tab()

    with tab_compare:
        render_compare_tab()

    with tab_browse:
        render_browse_tab(doc_type_for_qa)

    with tab_invoices:
        render_invoices_tab(invoices_df)

    with tab_admin:
        render_admin_tab()


if __name__ == "__main__":
    main()
