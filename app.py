import streamlit as st
import hashlib
import os
import pickle
import time
import io
import base64
import numpy as np
import faiss
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = "faiss_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-004"
LLM_MODEL = "gemini-2.0-flash"
TOP_K = 3

# ── Client ────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass
    if not api_key:
        st.error(
            "**GEMINI_API_KEY not set.**  \n"
            "Local: add it to your `.env` file.  \n"
            "Streamlit Cloud: go to **App Settings → Secrets** and add:  \n"
            "```\nGEMINI_API_KEY = \"your-key-here\"\n```"
        )
        st.stop()

    client = genai.Client(api_key=api_key)

    # Probe the API immediately so auth failures show a clear message
    try:
        client.models.embed_content(model=EMBEDDING_MODEL, contents="ping")
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        if status in (401, 403) or "API_KEY_INVALID" in str(exc) or "PERMISSION_DENIED" in str(exc):
            st.error(
                f"**API key rejected (HTTP {status}).**  \n"
                "The key in Streamlit Secrets does not match an active Gemini API key.  \n"
                "1. Go to [Google AI Studio](https://aistudio.google.com/apikey) and copy your key.  \n"
                "2. In Streamlit Cloud open **⋮ → Settings → Secrets** and update `GEMINI_API_KEY`.  \n"
                "3. Click **Save** — the app restarts automatically."
            )
            st.stop()
        # 429 on startup probe is fine — key is valid, just rate-limited
        if status not in (429,):
            st.error(f"**API connection error (HTTP {status}):** {type(exc).__name__}. Check the logs.")
            st.stop()

    return client

# ── Cache helpers ─────────────────────────────────────────────────────────────

def _files_hash(files_data: list[tuple[str, bytes]]) -> str:
    combined = b"".join(name.encode() + b for name, b in files_data)
    return hashlib.md5(combined).hexdigest()


def _cache_paths(file_hash: str):
    base = os.path.join(CACHE_DIR, file_hash)
    return base + ".faiss", base + ".pkl"


def _load_cache(file_hash: str):
    fp, pp = _cache_paths(file_hash)
    if os.path.exists(fp) and os.path.exists(pp):
        index = faiss.read_index(fp)
        with open(pp, "rb") as f:
            data = pickle.load(f)
        return index, data["chunks"], data["metadata"]
    return None, None, None


def _save_cache(file_hash: str, index, chunks, metadata):
    fp, pp = _cache_paths(file_hash)
    faiss.write_index(index, fp)
    with open(pp, "wb") as f:
        pickle.dump({"chunks": chunks, "metadata": metadata}, f)

# ── PDF processing ────────────────────────────────────────────────────────────

def extract_chunks(files_data: list[tuple[str, bytes]]):
    """Returns (chunks, metadata, errors). metadata items: {source, page}"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
    )
    chunks, metadata, errors = [], [], []

    for filename, file_bytes in files_data:
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            has_text = False
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    has_text = True
                    for chunk in splitter.split_text(text):
                        chunks.append(chunk)
                        metadata.append({"source": filename, "page": page_num})
            if not has_text:
                errors.append(
                    f"**{filename}** appears to be a scanned (image-only) PDF. "
                    "PyPDF is unable to extract text from images without an external OCR library. "
                    "Please ensure the PDF has embedded, searchable text, or run it through an "
                    "OCR tool before uploading."
                )
        except Exception as exc:
            errors.append(f"**{filename}**: {exc}")

    return chunks, metadata, errors

# ── Embedding / FAISS ─────────────────────────────────────────────────────────

def _embed_one(client, text: str) -> list[float]:
    """Embed a single text with retry/backoff for rate limits and transient errors."""
    for attempt in range(6):
        try:
            result = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
            return result.embeddings[0].values
        except Exception as exc:
            err = str(exc)
            is_rate  = "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower()
            is_server = "503" in err or "UNAVAILABLE" in err or "500" in err
            if attempt == 5:
                raise
            if is_rate:
                wait = 15 * (attempt + 1)   # 15 s, 30 s, 45 s …
                time.sleep(wait)
            elif is_server:
                time.sleep(10)
            else:
                raise   # auth / bad-request errors — no point retrying


def embed_texts(client, texts: list[str], progress=None) -> np.ndarray:
    vectors = []
    for i, text in enumerate(texts):
        vectors.append(_embed_one(client, text))
        if progress is not None:
            progress.progress((i + 1) / len(texts), text=f"Embedding {i+1}/{len(texts)} chunks…")
    return np.array(vectors, dtype=np.float32)


def build_index(client, chunks: list[str], progress=None):
    embeddings = embed_texts(client, chunks, progress=progress)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve(client, index, chunks, metadata, query: str, k: int = TOP_K):
    q_vec = embed_texts(client, [query])
    _, idxs = index.search(q_vec, k)
    return [
        {"text": chunks[i], "source": metadata[i]["source"], "page": metadata[i]["page"]}
        for i in idxs[0]
        if 0 <= i < len(chunks)
    ]

# ── LLM ──────────────────────────────────────────────────────────────────────

def ask_gemini(client, retrieved: list[dict], question: str, history: list[dict]) -> str:
    context = "\n\n---\n\n".join(r["text"] for r in retrieved)

    history_text = ""
    for turn in history[-8:]:  # last 4 exchanges
        role = "User" if turn["role"] == "user" else "Assistant"
        history_text += f"{role}: {turn['content']}\n"

    history_section = f"Previous conversation:\n{history_text}\n\n" if history_text else ""

    prompt = (
        "You are a helpful assistant answering questions about uploaded PDF documents.\n"
        "Answer ONLY using the context provided below.\n"
        "If the answer is not in the context, say: "
        "\"I don't know based on the given document.\"\n\n"
        f"{history_section}"
        f"Document Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    for attempt in range(4):
        try:
            response = client.models.generate_content(model=LLM_MODEL, contents=prompt)
            return response.text
        except Exception as exc:
            err = str(exc)
            is_rate   = "429" in err or "RESOURCE_EXHAUSTED" in err
            is_server = "503" in err or "UNAVAILABLE" in err
            if attempt == 3 or (not is_rate and not is_server):
                raise
            time.sleep(15 * (attempt + 1))

# ── PDF Viewer helper ─────────────────────────────────────────────────────────

def _pdf_iframe(pdf_bytes: bytes, page: int) -> str:
    b64 = base64.b64encode(pdf_bytes).decode()
    return (
        f'<iframe src="data:application/pdf;base64,{b64}#page={page}" '
        f'width="100%" height="420" '
        f'style="border:1px solid #e0e0e0; border-radius:6px;"></iframe>'
    )

# ── Session state defaults ────────────────────────────────────────────────────

_defaults = {
    "index": None,
    "chunks": [],
    "metadata": [],
    "messages": [],
    "pdf_names": [],
    "pdf_bytes_map": {},
    "cache_hit": None,
    "process_time": None,
    "extraction_errors": [],
    "processed_hash": None,
    "last_sources": [],
    "viewer_file": None,
    "viewer_page": 1,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")
client = get_client()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
    )

    if uploaded_files:
        # Read file bytes (getvalue() always returns full content)
        files_data = [(f.name, f.getvalue()) for f in uploaded_files]
        for name, b in files_data:
            st.session_state.pdf_bytes_map[name] = b

        current_hash = _files_hash(files_data)

        btn1, btn2 = st.columns(2)
        with btn1:
            process_clicked = st.button("Process", use_container_width=True, type="primary")
        with btn2:
            rebuild_clicked = st.button("Rebuild", use_container_width=True)

        # Auto-process if new files uploaded and index is empty
        auto_process = (
            current_hash != st.session_state.processed_hash
            and st.session_state.index is None
            and not process_clicked
            and not rebuild_clicked
        )

        if process_clicked or rebuild_clicked or auto_process:
            with st.spinner("Processing PDFs…"):
                t0 = time.perf_counter()

                # Try cache unless Rebuild was pressed
                cached_index, cached_chunks, cached_meta = (
                    (None, None, None) if rebuild_clicked
                    else _load_cache(current_hash)
                )

                if cached_index is not None:
                    st.session_state.index = cached_index
                    st.session_state.chunks = cached_chunks
                    st.session_state.metadata = cached_meta
                    st.session_state.cache_hit = True
                    st.session_state.extraction_errors = []
                else:
                    chunks, meta, errors = extract_chunks(files_data)
                    st.session_state.extraction_errors = errors

                    if chunks:
                        progress_bar = st.progress(0, text="Embedding chunks…")
                        try:
                            idx = build_index(client, chunks, progress=progress_bar)
                            progress_bar.empty()
                        except Exception as exc:
                            progress_bar.empty()
                            status = getattr(exc, "status_code", None)
                            st.error(
                                f"**Embedding failed (HTTP {status}, {type(exc).__name__}).**  \n"
                                "Most likely your API key in Streamlit Secrets is wrong or expired.  \n"
                                "Go to **⋮ → Settings → Secrets**, verify `GEMINI_API_KEY`, then **Save**."
                            )
                            st.session_state.processed_hash = None
                            st.stop()
                        _save_cache(current_hash, idx, chunks, meta)
                        st.session_state.index = idx
                        st.session_state.chunks = chunks
                        st.session_state.metadata = meta
                        st.session_state.cache_hit = False
                    else:
                        st.session_state.index = None
                        st.session_state.chunks = []
                        st.session_state.metadata = []

                st.session_state.process_time = time.perf_counter() - t0
                st.session_state.pdf_names = [f.name for f in uploaded_files]
                st.session_state.processed_hash = current_hash
                st.session_state.messages = []
                st.session_state.last_sources = []
                st.rerun()

        # ── Status ──────────────────────────────────────────────────────────
        if st.session_state.extraction_errors:
            for err in st.session_state.extraction_errors:
                st.error(f"**Text Extraction Failed**\n\n{err}")

        if st.session_state.index is not None:
            n = len(st.session_state.chunks)
            if st.session_state.cache_hit:
                st.success(f"Cache HIT — loaded in {st.session_state.process_time:.2f}s")
            else:
                st.success(f"Ready! {n} chunks indexed.")
            st.info(f"{n} chunks indexed")

            if st.button("Reprocess", use_container_width=True):
                st.session_state.index = None
                st.session_state.chunks = []
                st.session_state.metadata = []
                st.session_state.processed_hash = None
                st.rerun()

    st.divider()

    if st.session_state.index is not None:
        st.subheader("Document Info")
        for name in st.session_state.pdf_names:
            st.write(f"📄 {name}")
        st.write(f"**Chunks:** {len(st.session_state.chunks)}")

        st.subheader("Chunking Config")
        st.code(
            f"chunk_size    = {CHUNK_SIZE}\n"
            f"chunk_overlap = {CHUNK_OVERLAP}\n"
            f"top_k         = {TOP_K}\n"
            f"strategy      = RecursiveCharacter",
            language="text",
        )

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────

st.title("📄 PDF Chatbot")
st.caption(
    f"Chunk size: {CHUNK_SIZE} chars · Overlap: {CHUNK_OVERLAP} chars · "
    "FAISS vector index with file-hash cache"
)

if st.session_state.index is None and not st.session_state.extraction_errors:
    st.info("Upload a PDF in the sidebar to get started.")

elif st.session_state.extraction_errors and st.session_state.index is None:
    for err in st.session_state.extraction_errors:
        st.error(err)

else:
    # Two-column layout: chat (left) | sources + viewer (right)
    chat_col, source_col = st.columns([3, 2])

    with chat_col:
        # Render chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if user_input := st.chat_input("Ask a question about your PDFs…"):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    sources = retrieve(
                        client,
                        st.session_state.index,
                        st.session_state.chunks,
                        st.session_state.metadata,
                        user_input,
                    )

                    if sources:
                        st.session_state.last_sources = sources
                        st.session_state.viewer_file = sources[0]["source"]
                        st.session_state.viewer_page = sources[0]["page"]

                    answer = ask_gemini(
                        client,
                        sources,
                        user_input,
                        history=st.session_state.messages[:-1],
                    )

                st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()

    with source_col:
        st.subheader("SOURCES")

        if not st.session_state.last_sources:
            st.caption("Ask a question to see sources here.")
        else:
            for i, src in enumerate(st.session_state.last_sources, 1):
                label = f"Source {i}: {src['source']} (p.{src['page']})"
                with st.expander(label, expanded=True):
                    st.caption(f"{src['source']}, page {src['page']}")
                    preview = src["text"][:300] + ("…" if len(src["text"]) > 300 else "")
                    st.text(preview)

            st.divider()

            # ── PDF Viewer ───────────────────────────────────────────────────
            viewer_file = st.session_state.viewer_file
            viewer_page = st.session_state.viewer_page

            if viewer_file and viewer_file in st.session_state.pdf_bytes_map:
                # Unique (file, page) pairs from current sources, for navigation
                unique_pages: list[tuple[str, int]] = []
                seen = set()
                for s in st.session_state.last_sources:
                    key = (s["source"], s["page"])
                    if key not in seen:
                        seen.add(key)
                        unique_pages.append(key)

                current_idx = next(
                    (i for i, (f, p) in enumerate(unique_pages)
                     if f == viewer_file and p == viewer_page),
                    0,
                )

                nav1, nav2, nav3 = st.columns([3, 1, 1])
                with nav1:
                    st.markdown(f"**PDF Viewer** &nbsp;|&nbsp; Page {viewer_page}", unsafe_allow_html=True)
                with nav2:
                    if st.button("◀ Prev", disabled=current_idx == 0, use_container_width=True):
                        pf, pp = unique_pages[current_idx - 1]
                        st.session_state.viewer_file = pf
                        st.session_state.viewer_page = pp
                        st.rerun()
                with nav3:
                    if st.button("Next ▶", disabled=current_idx >= len(unique_pages) - 1, use_container_width=True):
                        nf, np_ = unique_pages[current_idx + 1]
                        st.session_state.viewer_file = nf
                        st.session_state.viewer_page = np_
                        st.rerun()

                pdf_bytes = st.session_state.pdf_bytes_map[viewer_file]
                st.markdown(_pdf_iframe(pdf_bytes, viewer_page), unsafe_allow_html=True)
