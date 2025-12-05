# scripts/build_index_all.py
"""
Build a unified Chroma index for ALL OCR'd documents:

- Invoices
- Resumes
- Contracts
- Others

Reads:
    data/ocr_texts/**/*.txt

Writes:
    index/chroma/   (ChromaDB persistent index)

Collection name:
    "documents"

Metadata stored per document:
    - source_file : filename only (e.g. "invoice_Aaron_123.txt")
    - rel_path    : path relative to project root (e.g. "data/ocr_texts/invoice_Aaron_123.txt")
    - doc_type    : "invoice" | "resume" | "contract" | "other"
    - text_len    : length of OCR text (characters)

Usage (from project root):

    (docenv) python scripts/build_index_all.py

Preconditions:
    - OCR already run and .txt files exist in data/ocr_texts
    - Chroma + sentence-transformers installed (requirements.txt)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
from chromadb.utils import embedding_functions

# ===================== PATHS & CONFIG =====================

BASE_DIR = Path(__file__).resolve().parent.parent
OCR_DIR = BASE_DIR / "data" / "ocr_texts"
INDEX_DIR = BASE_DIR / "index" / "chroma"

COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# For very large corpora you can tune this, but it's fine for 100s–1000s of docs
BATCH_SIZE = 200


# ===================== HELPERS =====================

def log(msg: str) -> None:
    """Simple stdout logger."""
    print(msg, flush=True)


def read_ocr_file(path: Path) -> str:
    """Read a single OCR text file safely."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        log(f"⚠️  Skipping {path.name}: error reading file ({e})")
        return ""


def guess_doc_type(path: Path, text: str) -> str:
    """
    Infer document type using:
      - parent folder name (if meaningful)
      - filename
      - OCR text content

    Returns one of: "invoice", "resume", "contract", "other"
    """
    fname = path.name.lower()
    t = text.lower()
    parent = path.parent.name.lower()

    # Strong signals from folder name
    if "invoice" in parent:
        return "invoice"
    if "resume" in parent or "cv" in parent:
        return "resume"
    if "contract" in parent or "agreements" in parent:
        return "contract"

    # Signals from filename
    if "invoice" in fname:
        return "invoice"
    if "resume" in fname or "cv" in fname:
        return "resume"
    if "contract" in fname or "agreement" in fname:
        return "contract"

    # Signals from text content
    if "invoice" in t or "amount due" in t or "bill to" in t:
        return "invoice"
    if "curriculum vitae" in t or "resume" in t or ("education" in t and "experience" in t):
        return "resume"
    if "this agreement" in t or "party a" in t or "party b" in t or "hereinafter" in t:
        return "contract"

    return "other"


def collect_documents() -> Tuple[List[str], List[str], List[Dict]]:
    """
    Walk data/ocr_texts and collect:
      - ids        : list[str] (doc ids used by Chroma)
      - documents  : list[str] (full OCR text)
      - metadatas  : list[dict] (source_file, rel_path, doc_type, text_len)
    """
    if not OCR_DIR.exists():
        log(f"❌ OCR folder not found: {OCR_DIR}")
        log("   Run: python ocr/run_ocr_pdfs.py")
        sys.exit(1)

    txt_files = sorted(OCR_DIR.rglob("*.txt"))
    if not txt_files:
        log(f"❌ No .txt files found in {OCR_DIR}")
        log("   Make sure OCR has produced .txt files.")
        sys.exit(1)

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict] = []

    skipped_empty = 0

    log("===============================================")
    log("   DocuMind Lite — Build Unified Chroma Index  ")
    log("===============================================")
    log(f"🔎 Found {len(txt_files)} OCR text files. Reading and classifying...")

    for path in txt_files:
        text = read_ocr_file(path)
        if not text.strip():
            skipped_empty += 1
            continue

        doc_type = guess_doc_type(path, text)
        rel_path = path.relative_to(BASE_DIR).as_posix()
        doc_id = path.stem  # unique per file

        ids.append(doc_id)
        documents.append(text)
        metadatas.append(
            {
                "source_file": path.name,
                "rel_path": rel_path,
                "doc_type": doc_type,
                "text_len": len(text),
            }
        )

    if not documents:
        log("❌ All OCR files were empty or unreadable. Nothing to index.")
        sys.exit(1)

    if skipped_empty:
        log(f"⚠️  Skipped {skipped_empty} empty OCR text files.")

    return ids, documents, metadatas


def init_collection():
    """
    Initialize Chroma client + collection.

    Strategy:
      - Always try to delete any existing collection with this name
      - Then create a fresh collection

    This avoids delete(where={}) / InternalError issues in newer Chroma.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    log(f"📁 Using Chroma index directory: {INDEX_DIR}")

    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Try to delete existing collection (if any)
    try:
        client.delete_collection(name=COLLECTION_NAME)
        log(f"🧹 Deleted existing collection '{COLLECTION_NAME}'.")
    except Exception:
        log(f"ℹ️ No existing collection named '{COLLECTION_NAME}' to delete.")

    # Create fresh collection
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn,
    )
    log(f"✅ Created fresh collection '{COLLECTION_NAME}'.")
    return collection


def batch_add_to_collection(collection, ids: List[str], documents: List[str], metadatas: List[Dict]) -> None:
    """Add documents to Chroma in batches to avoid memory issues."""
    total = len(documents)
    log(f"📦 Indexing {total} documents into collection '{COLLECTION_NAME}' (batch size={BATCH_SIZE})")

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_ids = ids[start:end]
        batch_docs = documents[start:end]
        batch_meta = metadatas[start:end]

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
        )
        log(f"   → Indexed {end}/{total} documents...")


def summarize_doc_types(metadatas: List[Dict]) -> None:
    """Print simple summary of document types and lengths."""
    counts: Dict[str, int] = {}
    total_chars = 0

    for m in metadatas:
        dt = m.get("doc_type", "other")
        counts[dt] = counts.get(dt, 0) + 1
        total_chars += m.get("text_len", 0)

    log("📊 Document counts by type:")
    for dt, cnt in sorted(counts.items(), key=lambda x: x[0]):
        log(f"   - {dt}: {cnt}")

    avg_len = total_chars / max(1, len(metadatas))
    log(f"🧾 Total characters indexed: {total_chars:,}")
    log(f"🧾 Average characters per document: {avg_len:,.0f}")


# ===================== MAIN =====================

def main() -> None:
    # 1) Collect documents
    ids, documents, metadatas = collect_documents()

    # 2) Init collection (delete old, create new)
    collection = init_collection()

    # 3) Add in batches
    batch_add_to_collection(collection, ids, documents, metadatas)

    # 4) Summary
    summarize_doc_types(metadatas)

    log("✅ Index build complete.")
    log("   You can now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
