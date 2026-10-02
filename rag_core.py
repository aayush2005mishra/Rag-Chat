"""Pure RAG logic: chunking, embeddings, vector store, and engine. No Streamlit imports.

Uses the native Google Gemini REST API (works with the new "AQ." auth keys, which are
sent in the `x-goog-api-key` header). Only `requests` and `numpy` are needed.
"""

import json
import os
import time
from typing import Dict, Generator, List, Optional

import numpy as np
import requests

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001")
# If GEMINI_CHAT_MODEL is set, only that model is used. Otherwise these are tried in order.
_CHAT_MODEL_ENV = os.environ.get("GEMINI_CHAT_MODEL")
CHAT_MODEL_CANDIDATES = (
    [_CHAT_MODEL_ENV]
    if _CHAT_MODEL_ENV
    else ["gemini-flash-latest", "gemini-3.5-flash", "gemini-2.5-flash", "gemini-3-flash-preview"]
)
CHAT_MODEL = CHAT_MODEL_CANDIDATES[0]

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer ONLY using the provided context. "
    "If the answer isn't in the context, say 'Not in the provided documents.' "
    "Cite sources like [1], [2]."
)

EMPTY_INDEX_MESSAGE = "Index some documents first."


# ---------------------------------------------------------------- Gemini REST helpers
class GeminiError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"Gemini API error {status}: {message}")
        self.status = status


def get_api_key(api_key: Optional[str] = None) -> str:
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise GeminiError(0, "GEMINI_API_KEY is not set")
    return key.strip()


def _headers(api_key: str) -> Dict[str, str]:
    return {"x-goog-api-key": api_key, "Content-Type": "application/json"}


def _post(url: str, api_key: str, payload: dict, stream: bool = False, tries: int = 5):
    """POST with retries on rate-limit (429) and temporary server errors (5xx)."""
    for attempt in range(tries):
        resp = requests.post(
            url, headers=_headers(api_key), json=payload, stream=stream, timeout=120
        )
        if resp.status_code == 200:
            return resp
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < tries - 1:
            resp.close()
            time.sleep(2 ** (attempt + 1))
            continue
        try:
            msg = resp.json().get("error", {}).get("message", resp.text)
        except Exception:
            msg = resp.text
        status = resp.status_code
        resp.close()
        raise GeminiError(status, msg)


def _extract_text(obj: dict) -> str:
    out = []
    for cand in obj.get("candidates", []) or []:
        for part in (cand.get("content", {}) or {}).get("parts", []) or []:
            if part.get("thought"):
                continue
            if part.get("text"):
                out.append(part["text"])
    return "".join(out)


# ---------------------------------------------------------------- chunking / embeddings
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    """Split text into overlapping character chunks, preferring natural break points."""
    text = (text or "").strip()
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size - 1))

    chunks: List[str] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            window_start = start + int(chunk_size * 0.6)
            brk = max(
                text.rfind("\n", window_start, end),
                text.rfind(". ", window_start, end),
                text.rfind(" ", window_start, end),
            )
            if brk > start:
                end = brk + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def embed_texts(
    texts: List[str],
    batch_size: int = 100,
    api_key: Optional[str] = None,
    model: str = EMBED_MODEL,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> List[List[float]]:
    """Embed texts in batches using Gemini embeddings (max 100 per request)."""
    if not texts:
        return []
    key = get_api_key(api_key)
    batch_size = max(1, min(batch_size, 100))
    url = f"{GEMINI_BASE_URL}/models/{model}:batchEmbedContents"
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        payload = {
            "requests": [
                {
                    "model": f"models/{model}",
                    "content": {"parts": [{"text": t}]},
                    "taskType": task_type,
                }
                for t in batch
            ]
        }
        resp = _post(url, key, payload)
        data = resp.json()
        resp.close()
        embeddings.extend(e["values"] for e in data["embeddings"])
    return embeddings


# ---------------------------------------------------------------- vector store
class VectorStore:
    """In-memory vector store using cosine similarity over normalized vectors."""

    def __init__(self) -> None:
        self.chunks: List[str] = []
        self.metadata: List[dict] = []
        self.vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / (norms + 1e-10)

    @property
    def size(self) -> int:
        return len(self.chunks)

    def add(self, chunks: List[str], vectors: List[List[float]], metadata: List[dict]) -> None:
        if not chunks:
            return
        if not (len(chunks) == len(vectors) == len(metadata)):
            raise ValueError("chunks, vectors, and metadata must have the same length")
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        arr = self._normalize(arr)
        if self.vectors.size == 0:
            self.vectors = arr
        else:
            self.vectors = np.vstack([self.vectors, arr])
        self.chunks.extend(chunks)
        self.metadata.extend(metadata)

    def search(self, query_vec: List[float], k: int = 4) -> List[Dict]:
        if self.size == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-10)
        scores = self.vectors @ q
        k = max(1, min(k, self.size))
        top_idx = np.argsort(-scores)[:k]
        return [
            {
                "chunk": self.chunks[i],
                "score": float(scores[i]),
                "metadata": self.metadata[i],
            }
            for i in top_idx
        ]

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            vectors=self.vectors,
            chunks=np.array(json.dumps(self.chunks)),
            metadata=np.array(json.dumps(self.metadata)),
        )

    def load(self, path: str) -> "VectorStore":
        if not os.path.exists(path) and os.path.exists(path + ".npz"):
            path = path + ".npz"
        data = np.load(path, allow_pickle=False)
        self.vectors = data["vectors"].astype(np.float32)
        self.chunks = json.loads(str(data["chunks"]))
        self.metadata = json.loads(str(data["metadata"]))
        return self

    def clear(self) -> None:
        self.chunks = []
        self.metadata = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)


# ---------------------------------------------------------------- engine
class RAGEngine:
    """Index documents, retrieve relevant chunks, and answer using only that context."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        chat_model: Optional[str] = None,
        embed_model: str = EMBED_MODEL,
        chunk_size: int = 500,
        overlap: int = 80,
    ) -> None:
        self.api_key = get_api_key(api_key)
        self.chat_models = [chat_model] if chat_model else list(CHAT_MODEL_CANDIDATES)
        self.embed_model = embed_model
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.store = VectorStore()

    @property
    def chat_model(self) -> str:
        return self.chat_models[0]

    def index_documents(self, docs: List[Dict[str, str]]) -> int:
        """docs = [{"text": str, "source": str}]. Returns number of chunks added."""
        all_chunks: List[str] = []
        all_meta: List[dict] = []
        for doc in docs:
            source = doc.get("source", "unknown")
            pieces = chunk_text(doc.get("text", ""), self.chunk_size, self.overlap)
            for idx, piece in enumerate(pieces):
                all_chunks.append(piece)
                all_meta.append({"source": source, "chunk_index": idx})
        if not all_chunks:
            return 0
        vectors = embed_texts(
            all_chunks, api_key=self.api_key, model=self.embed_model, task_type="RETRIEVAL_DOCUMENT"
        )
        self.store.add(all_chunks, vectors, all_meta)
        return len(all_chunks)

    def retrieve(self, query: str, k: int = 4) -> List[Dict]:
        if self.store.size == 0 or not query.strip():
            return []
        q_vec = embed_texts(
            [query], api_key=self.api_key, model=self.embed_model, task_type="RETRIEVAL_QUERY"
        )[0]
        return self.store.search(q_vec, k=k)

    @staticmethod
    def _build_payload(query: str, hits: List[Dict], temperature: float) -> dict:
        context = "\n\n".join(
            f"[{i}] (source: {h['metadata'].get('source', 'unknown')})\n{h['chunk']}"
            for i, h in enumerate(hits, start=1)
        )
        return {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [
                {"role": "user", "parts": [{"text": f"Context:\n{context}\n\nQuestion: {query}"}]}
            ],
            "generationConfig": {"temperature": temperature},
        }

    def _call(self, method: str, payload: dict, stream: bool = False):
        """Try each candidate chat model until one exists (skips 404 'model not found')."""
        last_err: Optional[GeminiError] = None
        for model in list(self.chat_models):
            suffix = "?alt=sse" if stream else ""
            url = f"{GEMINI_BASE_URL}/models/{model}:{method}{suffix}"
            try:
                resp = _post(url, self.api_key, payload, stream=stream)
            except GeminiError as e:
                if e.status == 404 and len(self.chat_models) > 1:
                    last_err = e
                    continue
                raise
            if model != self.chat_models[0]:  # remember the working model
                self.chat_models.remove(model)
                self.chat_models.insert(0, model)
            return resp
        raise last_err or GeminiError(404, "No usable Gemini chat model found")

    def answer(self, query: str, k: int = 4, temperature: float = 0.0) -> Dict:
        hits = self.retrieve(query, k=k)
        if not hits:
            return {"answer": EMPTY_INDEX_MESSAGE, "sources": []}
        resp = self._call("generateContent", self._build_payload(query, hits, temperature))
        text = _extract_text(resp.json())
        resp.close()
        return {"answer": text, "sources": hits}

    def stream_answer(
        self,
        query: str,
        k: int = 4,
        temperature: float = 0.0,
        hits: Optional[List[Dict]] = None,
    ) -> Generator[str, None, None]:
        """Yield answer tokens. Pass pre-retrieved `hits` to avoid a second retrieval,
        so the caller can display sources separately from the stream."""
        if hits is None:
            hits = self.retrieve(query, k=k)
        if not hits:
            yield EMPTY_INDEX_MESSAGE
            return
        resp = self._call(
            "streamGenerateContent", self._build_payload(query, hits, temperature), stream=True
        )
        resp.encoding = "utf-8"
        try:
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                text = _extract_text(obj)
                if text:
                    yield text
        finally:
            resp.close()
