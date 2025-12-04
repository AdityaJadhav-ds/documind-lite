# app.py — DocuMind Lite UI

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import streamlit as st
from openai import OpenAI

# ===================== CONFIG =====================

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "index" / "chroma"
COLLECTION_NAME = "invoices"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_CONTEXT_CHARS_PER_DOC = 1000
DEFAULT_MODEL = "gpt-4o-mini"

STRUCTURED_INVOICES_PATH = BASE_DIR / "data" / "structured" / "invoices.csv"

# ⚠️ IMPORTANT: DO NOT hardcode your real API key in code you push to GitHub.
# Set OPENAI_API_KEY in your environment, OR temporarily paste it here while testing.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""  # put "sk-..." here ONLY for local testing


# ===================== DATA HELPERS =====================

def load_invoices_df() -> Optional[pd.DataFrame]:
    """Load structured invoices CSV if it exists."""
    if STRUCTURED_INVOICES_PATH.exists():
        try:
            df = pd.read_csv(STRUCTURED_INVOICES_PATH)
            return df
        except Exception as e:
            st.sidebar.error(f"Error loading invoices.csv: {e}")
            return None
    return None


# ===================== RAG HELPERS =====================

def get_chroma_collection():
    """Return the Chroma collection for invoices."""
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


def retrieve_docs(collection, question: str, top_k: int = 3) -> List[dict]:
    """Retrieve top-k relevant documents for a question."""
    results = collection.query(query_texts=[question], n_results=top_k)
    if not results["documents"] or not results["documents"][0]:
        return []

    docs_raw = results["documents"][0]
    metas = results["metadatas"][0]
    ids = results["ids"][0]

    docs: List[dict] = []
    for doc_id, meta, text in zip(ids, metas, docs_raw):
        docs.append(
            {
                "id": doc_id,
                "source_file": meta.get("source_file", "unknown"),
                "text": text,
            }
        )
    return docs


def build_context(docs: List[dict]) -> str:
    """Build a compact context string from retrieved docs."""
    if not docs:
        return "No relevant invoices retrieved."

    blocks: List[str] = []
    for d in docs:
        snippet = d["text"][:MAX_CONTEXT_CHARS_PER_DOC]
        blocks.append(f"[ID: {d['id']} | file: {d['source_file']}]\n{snippet}")
    return "\n\n---\n\n".join(blocks)


def call_llm(question: str, context: str, model: str = DEFAULT_MODEL) -> str:
    """Call OpenAI chat completion to answer based on provided context."""
    if not OPENAI_API_KEY:
        return "❌ OpenAI API key is not set. Set OPENAI_API_KEY env var or update app.py."

    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = (
        "You are DocuMind Lite, an assistant that answers questions about invoices.\n"
        "- Use ONLY the information in the provided invoice texts.\n"
        "- If the answer is unclear or missing, say you are not sure.\n"
        "- Include invoice number, date, and total when possible.\n"
        "- If multiple invoices are relevant, summarize them clearly."
    )

    user_prompt = f"""
Question:
{question}

Relevant invoice texts:
{context}

Instructions:
- Answer in 2–6 sentences.
- Quote invoice numbers and totals.
- If multiple invoices match, summarize them.
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
        return f"❌ Error calling OpenAI API: {e}"

    return resp.choices[0].message.content or "(No answer returned.)"


# ===================== STREAMLIT APP =====================

def main() -> None:
    st.set_page_config(page_title="DocuMind Lite", page_icon="📄", layout="wide")

    st.title("📄 DocuMind Lite — Document Intelligence (Invoices MVP)")
    st.caption("Query your invoices using OCR + RAG + LLM. More document types coming soon.")

    invoices_df = load_invoices_df()

    # ---------- Sidebar ----------
    with st.sidebar:
        st.header("⚙️ Settings")

        top_k = st.slider("Top-k invoices to retrieve", 1, 10, 3)

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
                "OCR + indexing for new uploads will be wired in later steps. "
                "For now, Q&A uses pre-indexed invoices."
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
            st.info("invoices.csv not found. Run batch_extract_invoices.py first.")

    # ---------- Main: Q&A ----------
    st.subheader("💬 Ask a question about your invoices (RAG + LLM)")
    question = st.text_input(
        "Question",
        placeholder="e.g., What is the total amount on Aaron Smayling's invoice?",
    )

    ask_clicked = st.button("Ask")
    if ask_clicked and question.strip():
        try:
            collection = get_chroma_collection()
        except Exception as e:
            st.error(f"Error loading Chroma collection: {e}")
        else:
            with st.spinner("Retrieving relevant invoices..."):
                docs = retrieve_docs(collection, question, top_k=top_k)

            if not docs:
                st.warning("No relevant invoices found for this question.")
            else:
                context = build_context(docs)

                with st.spinner("Asking the LLM..."):
                    answer = call_llm(question, context)

                st.markdown("### ✅ Answer")
                st.write(answer)

                with st.expander("📎 Show context invoices"):
                    for d in docs:
                        st.markdown(f"**{d['id']} — {d['source_file']}**")
                        snippet = d["text"][:700]
                        st.text(snippet + ("..." if len(d["text"]) > 700 else ""))

    st.markdown("---")

    # ---------- Main: Structured Search ----------
    st.subheader("🔍 Search invoices (structured search)")

    if invoices_df is None:
        st.info("Structured search unavailable: invoices.csv not found.")
        return

    col1, col2 = st.columns(2)

    with col1:
        search_text = st.text_input(
            "Search by text (doc_id / file name)",
            placeholder="e.g., Aaron, 2012, 35876",
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

    # text filter on doc_id + source_file
    if search_text.strip():
        s = search_text.lower()
        mask = (
            filtered["doc_id"].astype(str).str.lower().str.contains(s)
            | filtered["source_file"].astype(str).str.lower().str.contains(s)
        )
        filtered = filtered[mask]

    # numeric filter on total_amount
    if "total_amount" in filtered.columns and min_total > 0:
        try:
            filtered["total_amount_num"] = (
                filtered["total_amount"]
                .astype(str)
                .str.replace(",", "", regex=False)
                .astype(float)
            )
            filtered = filtered[filtered["total_amount_num"] >= min_total]
        except Exception:
            st.warning("Could not convert total_amount to numeric for filtering.")

    st.write(f"Found **{len(filtered)}** matching invoices.")
    # hide helper column if present
    if "total_amount_num" in filtered.columns:
        filtered = filtered.drop(columns=["total_amount_num"])

    st.dataframe(filtered, use_container_width=True)


if __name__ == "__main__":
    main()
