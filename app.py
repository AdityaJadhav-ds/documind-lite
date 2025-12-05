# app.py — DocuMind Lite (Upgraded RAG App)
# Multi-document intelligence over invoices, resumes & contracts.

from __future__ import annotations

import os
import re
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from uuid import uuid4
from datetime import datetime

import chromadb
import pandas as pd
import streamlit as st
from chromadb.utils import embedding_functions
from openai import OpenAI

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


# ===================== OCR HELPERS =====================


def ocr_pdf(file_bytes: bytes) -> Tuple[bool, str]:
    """Run OCR on a PDF (all pages)."""
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
    except ImportError:
        return (
            False,
            "OCR dependencies missing. Install: `pip install pdf2image pytesseract` "
            "and install Tesseract on your system.",
        )

    try:
        pages = convert_from_bytes(file_bytes)
    except Exception as e:
        return False, f"Error converting PDF to images: {e}"

    texts: List[str] = []
    for img in pages:
        try:
            txt = pytesseract.image_to_string(img)
            texts.append(txt)
        except Exception:
            continue

    full_text = "\n\n".join(texts)
    return True, full_text


def ocr_image(file_bytes: bytes) -> Tuple[bool, str]:
    """Run OCR on an image (PNG/JPG)."""
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        return (
            False,
            "OCR dependencies missing. Install: `pip install pillow pytesseract` "
            "and install Tesseract on your system.",
        )

    try:
        img = Image.open(BytesIO(file_bytes))
        txt = pytesseract.image_to_string(img)
    except Exception as e:
        return False, f"Error during image OCR: {e}"

    return True, txt


def ocr_and_index_upload(
    uploaded_file, doc_type: str, collection
) -> Tuple[bool, str]:
    """
    OCR the uploaded file and add it to the Chroma collection.

    Returns (success, message).
    """
    ext = Path(uploaded_file.name).suffix.lower()
    file_bytes = uploaded_file.getvalue()

    if ext == ".pdf":
        ok, text = ocr_pdf(file_bytes)
    elif ext in {".png", ".jpg", ".jpeg"}:
        ok, text = ocr_image(file_bytes)
    else:
        return False, f"Unsupported file type: {ext}. Use PDF / PNG / JPG."

    if not ok:
        return False, text  # error message from OCR helper

    cleaned = clean_text(text)
    if len(cleaned) < 40:
        return False, "OCR text too short or empty. Check document quality."

    new_id = f"app_{uuid4().hex}"
    meta = {
        "doc_type": doc_type,
        "source_file": uploaded_file.name,
        "rel_path": f"uploads/{uploaded_file.name}",
        "text_len": len(cleaned),
        "uploaded_via": "app_upload",
        "uploaded_at": datetime.utcnow().isoformat(),
    }

    try:
        collection.add(
            ids=[new_id],
            documents=[cleaned],
            metadatas=[meta],
        )
    except Exception as e:
        return False, f"Failed to index document in Chroma: {e}"

    return True, f"Indexed `{uploaded_file.name}` as `{new_id}` (type: {doc_type})."


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


# ===================== CHROMA / RAG HELPERS =====================


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


def retrieve_docs(
    collection,
    question: str,
    top_k: int = 3,
    doc_type_filter: str = "all",
) -> List[dict]:
    """Retrieve top-k relevant documents from Chroma for a question, with optional doc_type filter."""
    query_kwargs: Dict[str, Any] = {
        "query_texts": [question],
        "n_results": top_k,
    }

    doc_type_filter = (doc_type_filter or "all").lower()
    if doc_type_filter != "all":
        query_kwargs["where"] = {"doc_type": doc_type_filter}

    results = collection.query(**query_kwargs)

    if not results.get("documents") or not results["documents"][0]:
        return []

    docs_raw = results["documents"][0]
    metas = results["metadatas"][0]
    ids = results["ids"][0]

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


def build_context(docs: List[dict]) -> str:
    """
    Build a safe, cleaned, PII-masked context string from retrieved docs.

    Each document is clearly separated and numbered so the model
    can reference them without confusion.
    """
    if not docs:
        return "No relevant documents retrieved."

    blocks: List[str] = []
    for i, d in enumerate(docs, start=1):
        raw = d["text"]
        cleaned = clean_text(raw)
        safe = mask_pii(cleaned)
        snippet = safe[:MAX_CONTEXT_CHARS_PER_DOC]

        blocks.append(
            f"=== DOCUMENT {i} ===\n"
            f"ID: {d['id']}\n"
            f"Type: {d['doc_type']}\n"
            f"File: {d['source_file']}\n"
            f"Text:\n{snippet}"
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
        "- If the document is a contract, focus on parties, dates, and obligations."
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
            include=["metadatas"],  # ids auto-returned
        )
        num_docs = len(results.get("ids", []))
    except Exception:
        return "I tried to count documents, but there was an error reading the index."

    return f"There are **{num_docs} {doc_type} document(s)** currently indexed in DocuMind Lite."


# ---------- Document metadata search (for UI table) ----------


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
        "include": ["metadatas"],  # "ids" is NOT allowed here
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
            ["invoice", "resume", "contract", "other"],
            index=0,
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
        if invoices_df is not None:
            st.write(f"Total invoices in CSV: **{len(invoices_df)}**")
            csv_bytes = invoices_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download invoices.csv",
                data=csv_bytes,
                file_name="invoices.csv",
                mime="text/csv",
            )
        else:
            st.info("`invoices.csv` not found. Run invoice extraction to enable this.")

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


# ===================== TABS =====================


def render_qa_tab(doc_type_for_qa: str, top_k: int) -> None:
    """Main RAG Q&A tab."""
    st.subheader("💬 Ask a question about your documents")

    strict_mode = st.toggle(
        "Strict mode (only answer from documents, no guessing)",
        value=True,
        help="Turn this OFF only if you want the model to infer slightly when context is weak.",
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
        # 1) Index-based count questions (no LLM)
        if not specific_mode:
            special_answer = handle_count_question(question, collection)
            if special_answer is not None:
                st.markdown("### ✅ Answer (from index metadata)")
                st.write(special_answer)
                st.session_state.qa_history.insert(
                    0, {"question": question, "answer": special_answer}
                )
                return

        # 2) Normal RAG flow
        with st.spinner("Retrieving relevant documents from index…"):
            if specific_mode and specific_doc_id:
                docs = retrieve_single_doc(collection, specific_doc_id)
            else:
                docs = retrieve_docs(
                    collection,
                    question,
                    top_k=top_k,
                    doc_type_filter=doc_type_for_qa,
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
            for d in docs:
                raw = d["text"]
                cleaned = clean_text(raw)
                safe = mask_pii(cleaned)
                snippet = safe[:700]
                st.markdown(
                    f"**[{d['doc_type'].upper()}] {d['id']} — {d['source_file']}**"
                )
                st.text(snippet + ("..." if len(safe) > 700 else ""))

    st.markdown("---")

    if st.session_state.qa_history:
        st.markdown("#### 🕒 Recent Q&A (local only)")
        for item in st.session_state.qa_history[:5]:
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {item['answer']}")
            st.markdown("---")


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
        "Upload → OCR → Index → RAG over invoices, resumes & contracts.\n"
        "Beginner-friendly, but designed like a real-world document AI tool."
    )

    invoices_df = load_invoices_df()
    doc_type_for_qa, top_k = sidebar_layout(invoices_df)

    tab_qa, tab_browse, tab_invoices, tab_admin = st.tabs(
        ["💬 Q&A", "🔍 Browse documents", "📊 Invoices (CSV)", "🛠 Admin / Analytics"]
    )

    with tab_qa:
        render_qa_tab(doc_type_for_qa, top_k)

    with tab_browse:
        render_browse_tab(doc_type_for_qa)

    with tab_invoices:
        render_invoices_tab(invoices_df)

    with tab_admin:
        render_admin_tab()


if __name__ == "__main__":
    main()
