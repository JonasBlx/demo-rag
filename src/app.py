"""Public web UI for the RAG demo.

Demo: everything is visible — answer, retrieved chunks + scores, the exact
context sent to the model, timings, tokens and estimated cost. Cost/abuse
guardrails: Haiku model, capped max_tokens, a per-question length cap, a
per-visitor monthly quota (by IP), and a global monthly budget guard.
"""

import logging
import os
import time
from collections import deque

import streamlit as st

from src import config
from src.rag import build_context, generate, load_vectorstore, retrieve

st.set_page_config(page_title="RAG · LangChain Docs", page_icon="🔗", layout="wide")


@st.cache_resource(show_spinner="Loading index…")
def get_vectorstore():
    # Cached once per process (loads the embedder + ChromaDB).
    return load_vectorstore()


def resolve_api_key() -> str | None:
    return st.session_state.get("byok") or os.getenv("ANTHROPIC_API_KEY")


_MONTH = 30 * 86400  # rolling 30-day window


@st.cache_resource
def _global_log() -> deque:
    """All accepted shared-key request timestamps (budget guard)."""
    return deque()


@st.cache_resource
def _user_log() -> dict:
    """client id (IP) -> deque of that visitor's request timestamps."""
    return {}


# In-memory and shared across all sessions in the container. It resets if the
# container restarts (HF Spaces sleep/redeploy) — the durable cost ceiling is the
# spend limit you set on the Anthropic console.


def _client_id() -> str:
    """Best-effort visitor id from the proxy's forwarded IP."""
    try:
        xff = st.context.headers.get("X-Forwarded-For", "")
    except Exception:
        xff = ""
    return xff.split(",")[0].strip() if xff else "unknown"


def _prune(dq: deque, now: float) -> deque:
    while dq and now - dq[0] > _MONTH:
        dq.popleft()
    return dq


def _usage(client_id: str) -> tuple[int, int]:
    """Return (global requests this month, this visitor's requests this month)."""
    now = time.time()
    g = _prune(_global_log(), now)
    u = _prune(_user_log().setdefault(client_id, deque()), now)
    return len(g), len(u)


def _record_request(client_id: str) -> None:
    now = time.time()
    _global_log().append(now)
    _user_log().setdefault(client_id, deque()).append(now)


# --- Sidebar: full configuration on display ----------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown(
        f"""
- **LLM**: `{config.ANTHROPIC_MODEL}`
- **max_tokens**: `{config.MAX_TOKENS}`
- **Embeddings**: `{config.EMBEDDING_BACKEND}` · `{config.LOCAL_EMBEDDING_MODEL}`
- **top-k**: `{config.RETRIEVAL_K}`

**Free-tier limits** (lifted with your own key)
- **Per visitor**: `{config.MAX_REQUESTS_PER_USER_PER_MONTH}` questions / month
- **Per question**: `{config.MAX_QUESTION_CHARS}` chars max
- **Demo budget**: `{config.MAX_REQUESTS_PER_MONTH}` questions / month (all visitors)
"""
    )
    with st.expander("🔑 Use your own API key (optional)"):
        st.text_input(
            "ANTHROPIC_API_KEY",
            type="password",
            key="byok",
            help="If set, your requests use your key instead of the demo's.",
        )
    st.markdown(f"[📂 Source code]({config.REPO_URL})")

st.title("🔗 Chat with the LangChain documentation")
st.caption("RAG grounded in the official docs · sourced answers · fully inspectable.")

# --- Session state -----------------------------------------------------------
st.session_state.setdefault("messages", [])

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Landing (first load only): a welcome message with clickable example questions.
EXAMPLE_QUESTIONS = [
    "What's the difference between a retriever and a vectorstore?",
    "How does retrieval work in a RAG pipeline?",
    "What is a knowledge base in LangChain?",
]
clicked = None
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "👋 Ask me anything about the **LangChain documentation**. "
            "I answer only from the docs and cite the sources.\n\n"
            "ℹ️ The docs are in English — **ask in English for the best results.**\n\n"
            "**Try one of these:**"
        )
        for ex in EXAMPLE_QUESTIONS:
            if st.button(ex, key=f"example::{ex}", use_container_width=True):
                clicked = ex

typed = st.chat_input("Ask a question about LangChain… (in English)")
question = typed or clicked
if not question:
    st.stop()

# --- Guardrails --------------------------------------------------------------
if not resolve_api_key():
    st.error(
        "No API key available. Set `ANTHROPIC_API_KEY` (host secret) "
        "or your own key in the sidebar."
    )
    st.stop()

# Abuse guardrails — enforced only on the shared demo key. Use your own key
# (sidebar) to lift them.
if not st.session_state.get("byok"):
    if len(question) > config.MAX_QUESTION_CHARS:
        st.warning(
            f"Question too long ({len(question)} characters; the limit is "
            f"{config.MAX_QUESTION_CHARS} on the shared demo). Shorten it, or use "
            "your own API key in the sidebar."
        )
        st.stop()
    global_month, user_month = _usage(_client_id())
    if global_month >= config.MAX_REQUESTS_PER_MONTH:
        st.warning(
            "🙏 This demo's monthly budget has been used up. It refreshes over the "
            "coming weeks — or use your own API key in the sidebar to continue now."
        )
        st.stop()
    if user_month >= config.MAX_REQUESTS_PER_USER_PER_MONTH:
        st.warning(
            f"You've used your {config.MAX_REQUESTS_PER_USER_PER_MONTH} free questions "
            "for this month — thanks for trying it! They refresh next month, or use "
            "your own API key in the sidebar to continue now."
        )
        st.stop()
    _record_request(_client_id())

st.session_state.messages.append({"role": "user", "content": question})
with st.chat_message("user"):
    st.markdown(question)

# --- RAG pipeline, step by step (timed for visibility) -----------------------
with st.chat_message("assistant"):
    try:
        t0 = time.perf_counter()
        scored = retrieve(get_vectorstore(), question, config.RETRIEVAL_K)
        t1 = time.perf_counter()
        context = build_context(scored)
        llm = config.get_llm(api_key=st.session_state.get("byok") or None)
        with st.spinner("Generating…"):
            reply = generate(llm, question, context)
        t2 = time.perf_counter()
    except Exception:  # boundary: degrade gracefully, don't leak internals
        logging.exception("RAG pipeline error")  # detail goes to server logs only
        st.error("Something went wrong while answering. Please try again in a moment.")
        st.stop()

    answer = reply.content if isinstance(reply.content, str) else str(reply.content)
    st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # --- Everything is inspectable ---
    with st.expander(f"📄 Sources ({len(scored)} chunks)"):
        for i, (doc, score) in enumerate(scored, 1):
            src = doc.metadata.get("source", "unknown")
            st.markdown(f"**{i}. score {score:.3f}** — [{src}]({src})")
            st.text(doc.page_content[:400] + ("…" if len(doc.page_content) > 400 else ""))

    with st.expander("🧩 Exact context sent to the model"):
        st.text(context)

    usage = getattr(reply, "usage_metadata", None) or {}
    tin, tout = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    pin, pout = config.PRICING.get(config.ANTHROPIC_MODEL, (0, 0))
    cost = tin / 1e6 * pin + tout / 1e6 * pout
    with st.expander("📊 Metrics"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Retrieval", f"{(t1 - t0) * 1000:.0f} ms")
        c2.metric("Generation", f"{(t2 - t1):.2f} s")
        c3.metric("Tokens (in/out)", f"{tin}/{tout}")
        st.caption(f"Estimated request cost: ${cost:.5f}")
