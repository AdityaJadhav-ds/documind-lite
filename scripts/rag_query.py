# scripts/rag_query.py
"""
RAG query script for DocuMind Lite.

Usage:
    python scripts/rag_query.py "What is the total amount on Aaron Smayling's invoice?"

Optional flags:
    -k / --top_k      → number of docs to retrieve (default: 3)
    -m / --model      → OpenAI model name (default: gpt-4o-mini)
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI, APIError, AuthenticationError, RateLimitError

# ---------- Configuration ----------

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "index" / "chroma"
COLLECTION_NAME = "invoices"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TOP_K = 3
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_CONTEXT_CHARS_PER_DOC = 1200


@dataclass
class RetrievedDoc:
    doc_id: str
    source_file: str
    text: str


# ---------- Utilities ----------

def get_api_key() -> str:
    """
    Fetch OpenAI API key from environment.
    This keeps your key out of the codebase and safe from GitHub leaks.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key:
        print(
            "\n❌ OPENAI_API_KEY is not set.\n"
            "   Please set it in your environment variables.\n\n"
            "   Examples:\n"
            "   • PowerShell:\n"
            "       $env:OPENAI_API_KEY = 'sk-xxxxx'\n\n"
            "   • CMD:\n"
            "       set OPENAI_API_KEY=sk-xxxxx\n\n"
            "   • Streamlit Cloud Secrets:\n"
            "       OPENAI_API_KEY = \"sk-xxxxx\"\n"
        )
        sys.exit(1)

    return api_key
 


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a RAG query over indexed invoices."
    )
    parser.add_argument(
        "question",
        type=str,
        help="Natural language question about your invoices.",
    )
    parser.add_argument(
        "-k",
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of most relevant documents to retrieve (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI chat model to use (default: {DEFAULT_MODEL}).",
    )
    return parser.parse_args()


def init_chroma() -> chromadb.Collection:
    """Initialize Chroma persistent client and return the invoices collection."""
    if not INDEX_DIR.exists():
        print(f"❌ Index directory not found: {INDEX_DIR}")
        print("   Did you run scripts/build_index.py ?")
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(INDEX_DIR))

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
    except Exception as e:
        print(f"❌ Failed to load Chroma collection '{COLLECTION_NAME}': {e}")
        print("   Did you run scripts/build_index.py and use the same name?")
        sys.exit(1)

    return collection


def retrieve_documents(
    collection: chromadb.Collection, question: str, top_k: int
) -> List[RetrievedDoc]:
    """Retrieve top-k relevant documents from Chroma."""
    results = collection.query(query_texts=[question], n_results=top_k)

    if not results["documents"] or not results["documents"][0]:
        print("⚠ No documents returned from index for this query.")
        return []

    docs_raw = results["documents"][0]
    metas = results["metadatas"][0]
    ids = results["ids"][0]

    docs: List[RetrievedDoc] = []
    for doc_id, meta, text in zip(ids, metas, docs_raw):
        docs.append(
            RetrievedDoc(
                doc_id=str(doc_id),
                source_file=str(meta.get("source_file", "unknown")),
                text=text,
            )
        )
    return docs


def build_context(docs: List[RetrievedDoc]) -> str:
    """Build a compact context block for the LLM."""
    if not docs:
        return "No relevant invoices were retrieved from the index."

    blocks: List[str] = []
    for doc in docs:
        snippet = doc.text[:MAX_CONTEXT_CHARS_PER_DOC]
        blocks.append(
            f"[ID: {doc.doc_id} | file: {doc.source_file}]\n{snippet}"
        )
    return "\n\n---\n\n".join(blocks)


def build_prompt(question: str, context: str) -> tuple[str, str]:
    """Return (system_prompt, user_prompt)."""
    system_prompt = (
        "You are DocuMind Lite, an assistant that answers questions about invoices.\n"
        "- Use ONLY the information in the provided invoice texts.\n"
        "- If the answer is unclear or missing, say you are not sure.\n"
        "- When possible, include invoice number, date, and total in your answer.\n"
        "- If multiple invoices are relevant, summarize them clearly."
    )

    user_prompt = f"""
Question:
{question}

Relevant invoice texts:
{context}

Instructions:
- Answer concisely (2–6 sentences).
- Quote OPENAI_API_KEY  values (invoice number, dates, totals) directly from the context.
- If you had to infer something, make that clear.
"""
    return system_prompt, user_prompt


def call_llm(model: str, system_prompt: str, user_prompt: str, api_key: str) -> str:
    """Call OpenAI chat completion and return the answer text."""
    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except AuthenticationError:
        print("❌ Authentication failed. Check your OPENAI_API_KEY.")
        sys.exit(1)
    except RateLimitError:
        print("❌ Rate limit reached. Try again later or reduce request frequency.")
        sys.exit(1)
    except APIError as e:
        print(f"❌ OpenAI API error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error calling OpenAI: {e}")
        sys.exit(1)

    choice = resp.choices[0].message.content
    return choice or "(No answer returned from model.)"


def main() -> None:
    args = parse_args()
    api_key = get_api_key()

    print(f"\n🔎 Question: {args.question}")
    print(f"📚 Retrieving top {args.top_k} invoices from Chroma index...\n")

    collection = init_chroma()
    docs = retrieve_documents(collection, args.question, args.top_k)

    if not docs:
        print("❌ No relevant documents found. Try rephrasing your question.")
        sys.exit(0)

    context = build_context(docs)
    system_prompt, user_prompt = build_prompt(args.question, context)

    answer = call_llm(args.model, system_prompt, user_prompt, api_key)

    print("💬 Answer:\n")
    print(answer)
    print("\n📎 Used documents:")
    for d in docs:
        print(f" - {d.doc_id} ({d.source_file})")


if __name__ == "__main__":
    main()
             