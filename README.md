# RAG Chat

A Streamlit RAG app (powered by the free Google Gemini API) that lets you upload or paste documents, ask questions, and get answers grounded only in your content, with sources and similarity scores.

## Install

```bash
pip install -r requirements.txt
```

Get a free key at https://aistudio.google.com/apikey (new `AQ.` keys work) and put it in the `.env` file:

```
GEMINI_API_KEY=your-key-here
```

## Run

```bash
streamlit run app.py
```

## How to Use

1. **Upload** `.txt`/`.pdf` files (or paste text) in the sidebar and click the index button.
2. **Ask** a question in the chat box.
3. **See sources**: expand "📎 Sources" under each answer to view the source name, similarity score, and chunk preview.

## Architecture

```
 ┌────────────┐   text    ┌─────────────┐  chunks  ┌───────────────────────┐
 │ Upload /   │ ────────▶ │ chunk_text  │ ───────▶ │ embed_texts           │
 │ Paste Text │           │ (500 / 80)  │          │ gemini-embedding-001│
 └────────────┘           └─────────────┘          └──────────┬────────────┘
                                                              │ vectors
                                                              ▼
 ┌────────────┐  query    ┌─────────────┐  top-K   ┌──────────────────────┐
 │ User       │ ────────▶ │ embed query │ ───────▶ │ VectorStore.search   │
 │ Question   │           └─────────────┘          │ (cosine similarity)  │
 └─────▲──────┘                                    └──────────┬───────────┘
       │                                                      │ chunks + scores
       │ streamed answer          ┌───────────────────────────▼────────────┐
       └──────────────────────────│ Gemini chat model (context-only prompt)      │
                                  └────────────────────────────────────────┘
```

## Files

- `app.py`: Streamlit UI only
- `rag_core.py`: chunking, embeddings, vector store, RAG engine (no Streamlit imports)
