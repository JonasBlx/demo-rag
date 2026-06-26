"""Centralized configuration.

A single source of truth for the LLM and the embedder, shared by ingestion,
the RAG chain and the UI. Everything goes through environment variables
(12-factor): no secret is hard-coded.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# --- Index ---
PERSIST_DIR = os.getenv("PERSIST_DIR", "vectorstore")

# --- LLM (Anthropic Claude) ---
# Haiku by default: the demo is public and paid for by the owner, so we favor a
# fast, cheap model. Override with ANTHROPIC_MODEL.
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))

# --- Retrieval ---
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))

# --- Embeddings (Anthropic has no embeddings endpoint) ---
# Questions are assumed to be in English, so an English model is the right fit.
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local")  # "local" | "voyage"
LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
VOYAGE_EMBEDDING_MODEL = os.getenv("VOYAGE_EMBEDDING_MODEL", "voyage-3.5")

# --- Cost/abuse guardrails (demo open to everyone) ---
MAX_QUESTIONS_PER_SESSION = int(os.getenv("MAX_QUESTIONS_PER_SESSION", "20"))

# Public pricing (USD / 1M tokens) — used only to show an estimated cost.
PRICING = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-8": (5.0, 25.0),
}

REPO_URL = "https://github.com/JonasBlx/demo-rag"


def get_embeddings():
    """Build the embedder per EMBEDDING_BACKEND. Must be identical at ingestion
    and retrieval, otherwise search fails silently."""
    if EMBEDDING_BACKEND == "voyage":
        from langchain_voyageai import VoyageAIEmbeddings

        return VoyageAIEmbeddings(model=VOYAGE_EMBEDDING_MODEL)

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)


def get_llm(api_key: str | None = None):
    """Build ChatAnthropic.

    We do NOT pass `temperature`: Opus 4.7/4.8 reject it (400 error) while Haiku
    accepts it — omitting it gives a single code path valid for every model
    (style/determinism is steered through the prompt).
    """
    from langchain_anthropic import ChatAnthropic

    kwargs = {"model": ANTHROPIC_MODEL, "max_tokens": MAX_TOKENS, "timeout": 60}
    if api_key:
        kwargs["api_key"] = api_key
    return ChatAnthropic(**kwargs)
