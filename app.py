# app.py — DocuMind Lite UI (Multi-Document RAG: invoices + resumes + contracts)

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

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
MAX_CONTEXT_CHARS_PER_DOC = 1000
DEFAULT_MODEL = "gpt-4o-mini"

STRUCTURED_INVOICES_PATH = BASE_DIR / "data" / "structured" / "invoices.csv"

# Read OpenAI API key from environment / Streamlit secret
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
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

def load_invoices_df() -> Optional[pd.DataFrame]:
    """Load structured invoices CSV if present."""
    if not STRUCTURED_INVOICES_PATH.exists():
        return None
    try:
        df = pd.read_csv(STRUCTURED_INVOICES_PATH)
        return df
    except Exception as e:
        st.sidebar.error(f"Error loading invoices.csv: {e}")
        return None


# ===================== RAG HELPERS =====================

def get_chroma_collection():
    """Return the unified Chroma collection for all documents."""
    if not INDEX_DIR.exists():
        raise RuntimeError(f"Index directory not found: {INDEX_DIR}")

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
    query_kwargs = {
        "query_texts": [question],
        "n_results": top_k,
    }

    if doc_type_filter and doc_type_filter.lower() != "all":
        query_kwargs["where"] = {"doc_type": doc_type_filter.lower()}

    results = collection.query(**query_kwargs)

    if not results["documents"] or not results["documents"][0]:
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


def build_context(docs: List[dict]) -> str:
    """Build a safe, cleaned, PII-masked context string from retrieved docs."""
    if not docs:
        return "No relevant documents retrieved."

    blocks: List[str] = []
    for d in docs:
        raw = d["text"]
        cleaned = clean_text(raw)
        safe = mask_pii(cleaned)
        snippet = safe[:MAX_CONTEXT_CHARS_PER_DOC]
        blocks.append(
            f"[ID: {d['id']} | type: {d['doc_type']} | file: {d['source_file']}]\n{snippet}"
        )

    return "\n\n---\n\n".join(blocks)


def call_llm(question: str, context: str, model: str = DEFAULT_MODEL) -> Tuple[bool, str]:
    """
    Call OpenAI chat completion using cleaned + masked context.

    Returns (success, message):
      - success=True  -> message is the answer
      - success=False -> message is an error description
    """
    if not HAS_API_KEY:
        return False, (
            "OPENAI_API_KEY is not set.\n"
            "Set it as:\n"
            "- Local PowerShell:  $env:OPENAI_API_KEY = 'sk-...'\n"
            "- Streamlit Cloud:   Settings → Secrets → OPENAI_API_KEY='sk-...'"
        )

    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = (
        "You are DocuMind Lite, an assistant that answers questions about documents.\n"
        "- Documents may be invoices, resumes, contracts, or others.\n"
        "- Use ONLY the information in the provided texts.\n"
        "- If the answer is unclear or missing, say you are not sure.\n"
        "- If the document is an invoice, include invoice number, date, and total when possible.\n"
        "- If the document is a resume, focus on skills, experience, and education.\n"
        "- If the document is a contract, focus on parties, dates, and obligations.\n"
        "- If multiple documents are relevant, summarize them clearly."
    )

    user_prompt = f"""
Question:
{question}

Relevant (PII-masked) documents:
{context}

Instructions:
- Answer in 2–8 sentences.
- Quote values directly from the context where possible.
- If multiple documents match, summarize them.
- If the answer cannot be found in the context, clearly say you are not sure.
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


# ===================== STREAMLIT APP =====================

def main() -> None:
    st.set_page_config(page_title="DocuMind Lite", page_icon="📄", layout="wide")

    st.title("📄 DocuMind Lite — Multi-Document Intelligence")
    st.caption(
        "OCR → Index → RAG over invoices, resumes, contracts. "
        "Invoices also have structured CSV + search."
    )

    invoices_df = load_invoices_df()

    # ---------- Sidebar ----------
    with st.sidebar:
        st.header("⚙️ Settings")

        doc_type_for_qa = st.selectbox(
            "Limit Q&A to document type",
            ["all", "invoice", "resume", "contract"],
            index=0,
        )

        top_k = st.slider("Top-k documents to retrieve for Q&A", 1, 10, 3)

        st.markdown("---")
        st.subheader("📤 Upload (stub for now)")
        uploaded = st.file_uploader(
            "Upload invoice/resume/contract (PDF)", type=["pdf"]
        )
        if uploaded is not None:
            upload_dir = BASE_DIR / "data" / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            save_path = upload_dir / uploaded.name
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.success(f"Saved to {save_path}")
            st.info(
                "OCR + indexing for new uploads will be wired later.\n"
                "Currently, Q&A uses your pre-indexed documents."
            )

        st.markdown("---")
        st.subheader("📊 Structured invoices")

        if invoices_df is not None:
            st.write(f"Total invoices: **{len(invoices_df)}**")

            csv_bytes = invoices_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download invoices.csv",
                data=csv_bytes,
                file_name="invoices.csv",
                mime="text/csv",
            )
        else:
            st.info("invoices.csv not found. Run invoice extraction to enable structured tools.")

        st.markdown("---")
        if HAS_API_KEY:
            st.success("OPENAI_API_KEY found ✅ — LLM Q&A is enabled.")
        else:
            st.error(
                "OPENAI_API_KEY is NOT set ❌\n\n"
                "Set it locally or as a Streamlit secret to enable LLM answers."
            )

    # ---------- Main: Q&A ----------
    st.subheader("💬 Ask a question about your documents (RAG + LLM)")
    question = st.text_input(
        "Question",
        placeholder=(
            "Examples:\n"
            "- What is the total amount on Aaron Smayling's invoice?\n"
            "- What are the main skills in this resume?\n"
            "- Who are the parties and effective date in the contract?"
        ),
    )

    ask_clicked = st.button("Ask", disabled=False)

    if ask_clicked and question.strip():
        try:
            collection = get_chroma_collection()
        except Exception as e:
            st.error(f"Error loading Chroma collection: {e}")
        else:
            with st.spinner("Retrieving relevant documents..."):
                docs = retrieve_docs(
                    collection,
                    question,
                    top_k=top_k,
                    doc_type_filter=doc_type_for_qa,
                )

            if not docs:
                st.warning(
                    "No relevant documents found for this question.\n"
                    "Tip: make sure OCR was run for resumes/contracts and "
                    "`scripts/build_index_all.py` was executed."
                )
            else:
                context = build_context(docs)

                st.markdown("---")
                success, answer = call_llm(question, context)

                if success:
                    st.markdown("### ✅ Answer")
                    st.write(answer)
                else:
                    st.markdown("### ❌ Could not get an answer")
                    st.error(answer)

                with st.expander("📎 Show context documents (masked text)"):
                    for d in docs:
                        raw = d["text"]
                        cleaned = clean_text(raw)
                        safe = mask_pii(cleaned)
                        snippet = safe[:700]
                        st.markdown(f"**[{d['doc_type'].upper()}] {d['id']} — {d['source_file']}**")
                        st.text(snippet + ("..." if len(safe) > 700 else ""))

    st.markdown("---")

    # ---------- Main: Structured Search (invoices only) ----------
    st.subheader("🔍 Search invoices (structured search)")

    if invoices_df is None:
        st.info("Structured search unavailable: invoices.csv not found.")
        return

    col1, col2 = st.columns(2)

    with col1:
        search_text = st.text_input(
            "Search by text (doc_id / file name)",
            placeholder="e.g., Aaron, 2012, 35876, .pdf",
            key="search_text",
        )

    with col2:
        min_total = st.number_input(
            "Minimum total amount (optional)",
            min_value=0.0,
            value=0.0,
            step=50.0,
            key="min_total",
        )

    filtered = invoices_df.copy()

    if search_text.strip():
        s = search_text.lower()
        mask = (
            filtered.get("doc_id", pd.Series("", index=filtered.index))
            .astype(str)
            .str.lower()
            .str.contains(s)
        ) | (
            filtered.get("source_file", pd.Series("", index=filtered.index))
            .astype(str)
            .str.lower()
            .str.contains(s)
        )
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

    st.write(f"Found **{len(filtered)}** matching invoices.")
    if "total_amount_num" in filtered.columns:
        filtered = filtered.drop(columns=["total_amount_num"])

    st.dataframe(filtered, use_container_width=True)


if __name__ == "__main__":
    main()
