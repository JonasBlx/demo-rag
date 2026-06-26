"""Ingestion pipeline: docs (markdown) → chunks → embeddings → ChromaDB.

Run once to build the index (executed during the Docker image build):
    python -m src.ingest

Uses only the embedder (local by default) — no API key required here.

The LangChain docs (docs.langchain.com) are a JS-rendered site: scraping the HTML
returns the same pre-rendered shell for every page. Instead we fetch the **markdown**
endpoint each page exposes (`<url>.md`) — clean, real content, no JavaScript needed.
A per-page diagnostic is printed so we can confirm every page produced content.
"""

from collections import Counter

import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import PERSIST_DIR, get_embeddings

BASE = "https://docs.langchain.com/oss/python/langchain"

# Pages relevant to a docs RAG. The retriever/vectorstore concepts now live in
# `retrieval`. Extend this list and re-ingest to broaden coverage.
PAGES = [
    "overview",
    "quickstart",
    "retrieval",
    "knowledge-base",
    "rag",
    "agents",
    "models",
    "tools",
]


def load_documents() -> list[Document]:
    docs = []
    for slug in PAGES:
        resp = requests.get(
            f"{BASE}/{slug}.md", timeout=30, headers={"User-Agent": "demo-rag/0.1"}
        )
        resp.raise_for_status()
        text = resp.text
        # Drop the shared "Documentation Index" preamble before the page's H1.
        h1 = text.find("\n# ")
        if h1 != -1:
            text = text[h1 + 1 :]
        # Cite the human-facing page URL (without the .md suffix).
        docs.append(Document(page_content=text, metadata={"source": f"{BASE}/{slug}"}))
    return docs


def main() -> None:
    docs = load_documents()

    print("Text fetched per page (detects empty pages):")
    for doc in docs:
        print(f"  {len(doc.page_content):>6} chars  {doc.metadata['source']}")
    print(f"[OK] {len(docs)} pages loaded")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    counts = Counter(c.metadata.get("source", "?") for c in chunks)
    print(f"\n{len(chunks)} chunks total:")
    for src, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>3} chunks  {src}")

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"},  # clean relevance scores
    )
    print(f"\n[OK] Index persisted in {PERSIST_DIR}/")


if __name__ == "__main__":
    main()
