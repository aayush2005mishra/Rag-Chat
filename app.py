"""Streamlit UI for the RAG app. All RAG logic lives in rag_core.py."""

import os
from pathlib import Path

import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv not installed; fall back to real environment variables

from rag_core import RAGEngine

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

st.set_page_config(page_title="RAG Chat", page_icon="📚", layout="wide")

if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
    st.error(
        "GEMINI_API_KEY is not set. Put it in the .env file "
        "(GEMINI_API_KEY=your-key) and restart the app. "
        "Get a free key at https://aistudio.google.com/apikey"
    )
    st.stop()

# ---------- Session state ----------
if "engine" not in st.session_state:
    st.session_state.engine = RAGEngine()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

engine: RAGEngine = st.session_state.engine


# ---------- Helpers ----------
def read_uploaded_file(uploaded) -> str:
    name = uploaded.name.lower()
    if name.endswith(".pdf"):
        if PdfReader is None:
            st.warning(f"Skipping {uploaded.name}: pypdf is not installed (pip install pypdf).")
            return ""
        reader = PdfReader(uploaded)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return uploaded.getvalue().decode("utf-8", errors="ignore")


def register_sources(names):
    for n in names:
        if n not in st.session_state.indexed_docs:
            st.session_state.indexed_docs.append(n)


def render_sources(sources):
    with st.expander("📎 Sources"):
        for i, s in enumerate(sources, start=1):
            src = s["metadata"].get("source", "unknown")
            st.markdown(f"**[{i}] {src}** — similarity: `{s['score']:.3f}`")
            preview = s["chunk"]
            st.caption(preview[:300] + ("…" if len(preview) > 300 else ""))


# ---------- Sidebar ----------
with st.sidebar:
    st.header("📚 Knowledge Base")
    tab_upload, tab_paste = st.tabs(["📄 Upload Files", "✏️ Paste Text"])

    with tab_upload:
        files = st.file_uploader(
            "Upload .txt or .pdf files",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
        )
        if st.button("Index Uploaded Files", use_container_width=True):
            if not files:
                st.warning("Please upload at least one file.")
            else:
                docs = []
                for f in files:
                    try:
                        text = read_uploaded_file(f)
                    except Exception as e:
                        st.error(f"Could not read {f.name}: {e}")
                        continue
                    if text.strip():
                        docs.append({"text": text, "source": f.name})
                    else:
                        st.warning(f"No extractable text in {f.name}.")
                if docs:
                    try:
                        with st.spinner("Chunking and embedding..."):
                            n = engine.index_documents(docs)
                        register_sources(d["source"] for d in docs)
                        st.success(f"Indexed {n} chunks from {len(docs)} file(s).")
                    except Exception as e:
                        st.error(f"Indexing failed: {e}")

    with tab_paste:
        source_name = st.text_input("Source name", value="pasted-text")
        pasted = st.text_area("Paste content", height=200)
        if st.button("Index Pasted Text", use_container_width=True):
            if not pasted.strip():
                st.warning("Please paste some text.")
            else:
                try:
                    with st.spinner("Chunking and embedding..."):
                        n = engine.index_documents(
                            [{"text": pasted, "source": source_name.strip() or "pasted-text"}]
                        )
                    register_sources([source_name.strip() or "pasted-text"])
                    st.success(f"Indexed {n} chunks.")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    st.divider()
    st.metric("Total chunks indexed", engine.store.size)
    with st.expander("Indexed sources"):
        if st.session_state.indexed_docs:
            for name in st.session_state.indexed_docs:
                st.write(f"• {name}")
        else:
            st.caption("Nothing indexed yet.")

    if st.button("🗑️ Clear All", use_container_width=True):
        engine.store.clear()
        st.session_state.indexed_docs = []
        st.session_state.messages = []
        st.session_state.uploader_key += 1
        st.rerun()

    st.divider()
    st.subheader("Settings")
    top_k = st.slider("Top-K", min_value=1, max_value=10, value=4)
    show_sources = st.toggle("Show sources", value=True)

# ---------- Main ----------
st.title("💬 Ask Your Documents")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            render_sources(msg["sources"])

query = st.chat_input("Ask a question about your documents...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        if engine.store.size == 0:
            msg = "Index some documents first."
            st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg, "sources": []})
        else:
            try:
                hits = engine.retrieve(query, k=top_k)  # sources retrieved separately
                placeholder = st.empty()
                full_text = ""
                for token in engine.stream_answer(query, k=top_k, hits=hits):
                    full_text += token
                    placeholder.markdown(full_text + "▌")
                placeholder.markdown(full_text)
                if show_sources and hits:
                    render_sources(hits)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_text, "sources": hits}
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")
