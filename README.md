# Rag-Chat
RAG chat app built with Streamlit and Google Gemini. Upload PDFs/text, ask questions, get answers with sources and similarity scores.
RAG Chat: Ask Your Documents

A Retrieval-Augmented Generation (RAG) web app built with Python and Streamlit. Users upload .txt or .pdf files, or paste text, and then ask questions in a chat interface. The app answers using only the content of those documents, and shows which parts of the documents it used.

How it works

Documents are split into overlapping chunks (500 characters, 80 overlap).
Each chunk is converted to an embedding with Google's gemini-embedding-001 model.
Vectors are normalized and stored in memory, and a question is matched to the top-K chunks with cosine similarity (NumPy).
The matching chunks are sent to a Gemini chat model with a strict prompt: answer only from the context, otherwise say "Not in the provided documents."
The answer streams token by token, with a Sources panel showing the file name, similarity score and a chunk preview.

Features

PDF and text upload, plus pasted text
Streaming answers with source citations like [1], [2]
Adjustable Top-K (1-10) and a toggle to show or hide sources
Chunk counter, list of indexed sources, and a Clear All button
Retries when the API rate limit is hit, and an automatic fallback if a Gemini model name is unavailable
RAG logic kept separate from the UI (rag_core.py has no Streamlit imports), so it can be reused

