# PDF Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload a PDF and ask questions about its content using Google Gemini AI.

## Features

- **PDF Upload & Parsing** — Upload any PDF file; text is extracted from all pages automatically.
- **Smart Text Chunking** — Documents are split into overlapping chunks (1000 chars, 200 overlap) using LangChain's `RecursiveCharacterTextSplitter` for better context preservation.
- **Vector Embeddings** — Chunks are embedded using Google's `text-embedding-004` model and stored in a FAISS index for fast similarity search.
- **FAISS-Powered Retrieval** — Top-4 most relevant chunks are retrieved per query using L2 distance search.
- **Gemini LLM Answers** — Retrieved context is passed to `gemini-2.0-flash` to generate accurate, document-grounded answers.
- **File Hash Caching** — FAISS indexes are cached by MD5 hash of the PDF, so re-uploading the same file skips re-indexing instantly.
- **Chat History** — Maintains a multi-turn conversation history within the session.
- **Retrieved Chunk Inspector** — Expandable section shows the top 4 retrieved source chunks for every answer.
- **Rate Limit & Retry Handling** — Automatic retry logic with backoff for API rate limits (429) and server errors (503).
- **Streamlit UI** — Clean web interface with a sidebar for upload/document info and a main chat area.

## Tech Stack

| Component | Library |
|---|---|
| UI | Streamlit |
| LLM & Embeddings | Google Gemini (`google-genai`) |
| Vector Store | FAISS (`faiss-cpu`) |
| PDF Parsing | pypdf |
| Text Splitting | LangChain Text Splitters |
| Numerical ops | NumPy |

## Setup

1. **Clone the repo and create a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your Gemini API key** in a `.env` file:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Usage

1. Upload a PDF using the sidebar.
2. Wait for the index to build (or load instantly from cache).
3. Type your question in the chat input.
4. View the answer and optionally expand the "Retrieved chunks" section to see what the model used.

## Project Structure

```
pdf-chatbot/
├── app.py              # Main Streamlit app
├── rag_chain.py        # RAG pipeline (retrieve + generate)
├── build_index.py      # Script to pre-build FAISS index from PDFs
├── utils.py            # Shared utilities
├── faiss_cache/        # Cached FAISS indexes (keyed by file hash)
├── faiss_index/        # Pre-built index for batch-loaded PDFs
├── data/               # Sample PDF files
└── requirements.txt
```
