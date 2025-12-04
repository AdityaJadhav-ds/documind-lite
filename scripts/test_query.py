# scripts/test_query.py

from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "index" / "chroma"

client = chromadb.PersistentClient(path=str(INDEX_DIR))
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client.get_collection(
    name="invoices",
    embedding_function=embedding_fn,
)

query = "find invoices with total amount around 300"
results = collection.query(query_texts=[query], n_results=3)

print("Query:", query)
print("Top matches:")
for doc_id, meta in zip(results["ids"][0], results["metadatas"][0]):
    print(" -", doc_id, "->", meta["source_file"])
