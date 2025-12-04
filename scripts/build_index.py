# scripts/build_index.py

from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = Path(__file__).resolve().parent.parent
OCR_DIR = BASE_DIR / "data" / "ocr_texts"
INDEX_DIR = BASE_DIR / "index" / "chroma"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# 1) Create Chroma client (local persistent)
client = chromadb.PersistentClient(path=str(INDEX_DIR))

# 2) Use a small sentence-transformers model
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="invoices",
    embedding_function=embedding_fn,
)

# 3) Load all invoice txt files
docs = []
ids = []
metadatas = []

for txt_path in OCR_DIR.glob("invoice*.txt"):
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        continue

    doc_id = txt_path.stem  # e.g., invoice_Aaron Smayling_35876
    ids.append(doc_id)
    docs.append(text)
    metadatas.append({"source_file": txt_path.name})

# 4) Add to Chroma collection
if docs:
    collection.upsert(documents=docs, ids=ids, metadatas=metadatas)
    print(f"Indexed {len(docs)} invoices into Chroma.")
else:
    print("No invoice txt files found to index.")
