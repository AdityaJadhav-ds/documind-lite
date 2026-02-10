ddffds# 📄 DocuMind Lite — Multi-Document Intelligence  
### Upload → OCR → Index → Hybrid RAG → LLM → Insights  
**Created using ChatGPT (GPT-5)**  

DocuMind Lite is a production-inspired, lightweight document intelligence platform that processes **invoices, resumes, and contracts**.  
It performs OCR, document classification, structured extraction, semantic indexing, hybrid retrieval, and LLM-powered Q&A — all in a beautiful Streamlit application.

---

# 🌐 Live Demo  
👉 **https://docs-brain.streamlit.app/**

---

## 🧠 System Pipeline Architecture

A high-level overview of how **DocuMind Lite** processes documents end-to-end — from upload → OCR → indexing → hybrid retrieval → LLM reasoning.

<p align="center">
  <img src="DocuMind-Lite-System-Pipeline.png " alt="DocuMind Lite System Pipeline" width="90%">
</p>



---

# 🚀 Key Features

### ✔ **High-accuracy OCR Engine**
- PDF → Image → OCR → Clean text  
- Tesseract with noise removal  
- Auto PII masking (emails, phone numbers)

### ✔ **Automatic Document Classification**
- Invoice  
- Resume  
- Contract  
- Auto-detect mode

### ✔ **Structured Field Extraction**
- Invoices → invoice number, date, total  
- Resumes → skills, education, experience  
- Contracts → parties, obligations, dates  

### ✔ **Hybrid Retrieval (Vector + BM25)**
- SentenceTransformer embeddings  
- BM25 keyword search  
- Query-type classifier dynamically adjusts weighting  
- Optional LLM-based reranker for high-precision relevance

### ✔ **LLM-Powered Q&A Engine**
- Strict mode (zero hallucination)  
- Document citations (DOC1, DOC2…)  
- Builds clean, PII-masked context  
- Multi-document reasoning

### ✔ **Professional UI / UX**
- Q&A Assistant  
- PDF/Image Viewer  
- Document Comparison  
- Document Browser  
- Structured Invoice Explorer  
- Admin Dashboard & Analytics  

---

# 📦 Installation (Local, No Docker)

bash
git clone <https://github.com/AdityaJadhav-ds/documind-lite.git>
cd documind-lite

python -m venv .venv
.\.venv\Scripts\activate        # Windows
# OR
source .venv/bin/activate       # macOS/Linux

pip install -r requirements.txt
Run the app:

bash
Copy code
streamlit run app.py
Open in browser:
👉 http://localhost:8501

🔑 Set Your API Key
Option A — Set ENV variable (recommended)
Windows (PowerShell)

powershell
Copy code
$env:OPENAI_API_KEY="sk-..."
macOS / Linux

bash
Copy code
export OPENAI_API_KEY="sk-..."
Option B — Streamlit Secrets
Create:

/.streamlit/secrets.toml

toml
Copy code
OPENAI_API_KEY = "sk-..."

---

## 📁 Project Structure
bash
Copy code
documind-lite/
│
├── app.py                         # Main Application
├── ocr_engine.py                  # OCR + preprocessing
├── keyword_search.py              # BM25 search engine
├── reranker.py                    # LLM embedding reranker
├── invoice_extractor.py           # Structured invoice extraction
├── resume_contract_extractor.py   # Resume + contract extraction
│
├── uploads/                       # Uploaded PDFs/images
├── index/                         # Chroma vector database
├── data/
│   └── structured/invoices.csv    # Structured output
│
├── requirements.txt
└── README.md

---

### 🛠 Troubleshooting Guide
❌ Incorrect API Key
Double-check API key from https://platform.openai.com.
Make sure no extra spaces were pasted.

❌ Chroma _type Error
Delete corrupted index:

bash
Copy code
rm -rf index/
Then restart the app — index rebuilds automatically.

❌ set_page_config() Error
Make sure this is the first Streamlit command in app.py.

❌ Session State Error
Initialize session keys before referencing them.

🧭 Roadmap
Local/offline embeddings

Resume professional summarizer

Contract clause extractor

Bulk document ingestion (cloud storage)

Integrations: Notion / Airtable / PostgreSQL

---

# 🙌 Credits

This project was developed with guided support from ChatGPT (GPT-5) acting as:

Architect

ML Engineer

Prompt Engineer

UX Designer

Debugging Assistant

