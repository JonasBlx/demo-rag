# Objectives — RAG over the LangChain documentation

Refined specification of the project. Describes **what we build, why, and with which justified
technical choices**. The code snippets below are **design references** (not the final code); the
implementation lives in `src/` and follows [`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## Goal

Query ~500 pages of technical documentation (LangChain) in natural language through a public
chatbot that answers **with source citations** and **refuses to make things up** when the
information is not in the retrieved context. Questions are assumed to be in **English**.

Success criteria (measurable):

1. A factual question covered by the docs gets a correct, sourced answer.
2. An off-topic question gets an explicit refusal, no hallucination.
3. The full pipeline runs publicly with a single API key (`ANTHROPIC_API_KEY`).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        INGESTION                            │
│  LangChain docs (HTML) → chunking → embedding → ChromaDB    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                        RETRIEVAL                            │
│  question → embedding → cosine similarity → top-k chunks    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                        GENERATION                           │
│  chunks + question → prompt → Claude → answer with sources  │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech stack

| Component       | Chosen tool                           | Alternatives            |
|-----------------|---------------------------------------|-------------------------|
| Orchestration   | LangChain (LCEL)                      | LlamaIndex              |
| LLM             | Anthropic **Claude Haiku 4.5**        | `claude-opus-4-8` (quality) |
| Embedding       | **sentence-transformers** (local)     | Voyage AI (hosted)      |
| Vector store    | ChromaDB (local, persisted)           | FAISS, pgvector         |
| UI              | Streamlit                             | Gradio                  |
| Hosting         | Hugging Face Spaces (Docker)          | Render / Fly.io / Cloud Run |
| Doc loading     | `requests` (markdown `.md` endpoints) | `WebBaseLoader` (HTML)  |

### Why two models (generation ≠ embeddings)

**Anthropic does not provide an embeddings endpoint.** A Claude-based RAG must therefore get its
vectors elsewhere. Two options, selectable via `EMBEDDING_BACKEND` in `.env`:

- **`local` (default)** — `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface`.
  Free, no extra key, runs on CPU. Since questions are in English, an English embedder is the
  right fit. For more quality: `BAAI/bge-small-en-v1.5`.
- **`voyage`** — Voyage AI (`voyage-3.5`), the embedding partner **recommended by Anthropic**.
  Better retrieval quality, but requires `VOYAGE_API_KEY`.

> ⚠️ **Index/query consistency.** The same embedder must be used at ingestion *and* retrieval.
> Switching backend (or model) requires **rebuilding** the `vectorstore/`.

### Choice of Claude model

`claude-haiku-4-5` by default: the demo is **public and owner-paid**, so we favor a fast, cheap
model. For more quality: `claude-opus-4-8` (via `ANTHROPIC_MODEL`).

> ⚠️ **Opus 4.7/4.8 gotcha.** The `temperature`, `top_p`, `top_k` parameters are **rejected**
> (400 error) by Opus 4.7/4.8; Haiku accepts them. To keep **a single code path valid for every
> model**, `src/config.py` builds `ChatAnthropic` **without** `temperature` (style/determinism is
> steered through the prompt).

### Deployment & guardrails (demo open to all)

Hosted on **Hugging Face Spaces (Docker SDK)**, reachable by anyone without installation. The
`Dockerfile` builds the index at build time and bakes it into the image (stateless container, fast
startup). The API key is a **host secret**, never in the repo.

Since the endpoint is open and each call costs money, we bound the exposure:

- **`claude-haiku-4-5`** (low unit cost);
- capped **`MAX_TOKENS`**;
- **per-session question limit** (`MAX_QUESTIONS_PER_SESSION`);
- **BYO key** option: a visitor can paste their own key (cost/abuse transferred).

### Visibility ("everything must be visible")

The UI exposes every RAG step, because this is a teaching demo:

- the **retrieved chunks** with their **relevance score** and source URL;
- the **exact context** sent to the model;
- the **timings** (retrieval / generation) and **token usage** + estimated cost;
- the active **configuration** (model, k, embedder) in the sidebar;
- a link to the **source code** (public repo).

---

## Prerequisites

- Python 3.10+
- `ANTHROPIC_API_KEY` (https://console.anthropic.com/)
- *(Optional)* `VOYAGE_API_KEY` if `EMBEDDING_BACKEND=voyage`

Installation: see [`README.md`](README.md).

---

## Step 1 — Ingestion (design reference)

The LangChain docs live at `docs.langchain.com`. The site is **JS-rendered**, so scraping the HTML
returns the same pre-rendered shell for every page. Instead we fetch each page's **markdown**
endpoint (`<url>.md`) — clean, real content, no JavaScript — then chunk, embed, and persist.

```python
# src/ingest.py (reference)
import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from src.config import get_embeddings, PERSIST_DIR

BASE = "https://docs.langchain.com/oss/python/langchain"
PAGES = ["overview", "quickstart", "retrieval", "knowledge-base", "rag", "agents", "models", "tools"]

docs = []
for slug in PAGES:
    text = requests.get(f"{BASE}/{slug}.md", timeout=30).text
    docs.append(Document(page_content=text, metadata={"source": f"{BASE}/{slug}"}))

chunks = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # ~good context/precision tradeoff for technical docs
    chunk_overlap=150,      # avoid cutting a point across two chunks
    separators=["\n\n", "\n", ". ", " ", ""],
).split_documents(docs)

Chroma.from_documents(
    documents=chunks,
    embedding=get_embeddings(),
    persist_directory=PERSIST_DIR,            # automatic persistence (no .persist() call)
    collection_metadata={"hnsw:space": "cosine"},
)
```

> Ingestion runs once; the index is persisted in `./vectorstore/`. To add documents, re-run with
> new URLs (keeping the same embedder).

---

## Step 2 — RAG chain (design reference)

```python
# src/rag.py (reference)
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from src.config import get_llm, get_embeddings, PERSIST_DIR, RETRIEVAL_K

PROMPT = ChatPromptTemplate.from_template(
    "You are an expert assistant for LangChain. Answer using ONLY the context below. "
    "If the information is not in the context, say so explicitly — never make it up. "
    "Cite the sources (URLs) you rely on.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)

def load_vectorstore():
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=get_embeddings())

def retrieve(vs, question, k=RETRIEVAL_K):
    return vs.similarity_search_with_relevance_scores(question, k=k)  # (doc, score 0–1)

# get_llm() builds ChatAnthropic WITHOUT temperature (see Opus 4.8 gotcha).
```

The chain is kept **decomposed** (retrieve / build context / generate) on purpose, so the UI can
display each intermediate step.

---

## Step 3 — Streamlit UI (design reference)

Chat with history + sources shown under each answer. `@st.cache_resource` avoids reloading the
LLM/vectorstore on every interaction. See `src/app.py` for the implementation.

Key UI points:
- show Claude's answer;
- under expanders, list each retrieved chunk (URL + score + excerpt) and the exact context sent;
- show metrics (timings, tokens, estimated cost);
- keep history in `st.session_state`.

---

## Example questions to try

```
# Concepts
"What's the difference between a retriever and a vectorstore in LangChain?"
"How does the RecursiveCharacterTextSplitter work?"

# Practical
"How do I implement a RAG with cited sources?"
"Which retrieval strategies are available?"

# Off-topic (should be refused cleanly)
"What is the capital of Australia?"
```

---

## Possible improvements (out of initial scope)

1. **Hybrid search** semantic + BM25 via `EnsembleRetriever`.
2. **RAG evaluation** (faithfulness, answer relevancy, context recall) with RAGAS.
3. **Conversational memory**: rewrite the question from history before retrieval.
4. **Streaming** the Claude answer in the UI (`.stream()`), snappier on long answers.
5. **Voyage AI embeddings** for higher retrieval quality (`EMBEDDING_BACKEND=voyage`).

---

## What this project demonstrates

| Skill                 | Concrete evidence                                                      |
|-----------------------|------------------------------------------------------------------------|
| RAG architecture      | Full pipeline: ingestion → chunking → embedding → retrieval → generation |
| Chunking strategy     | Justified `chunk_size` / `overlap` choice                              |
| Vector store          | ChromaDB persisted locally                                             |
| Prompt engineering    | Context-constrained prompt, with sourcing and refusal on missing info  |
| LangChain             | LCEL, modern split packages                                            |
| LLM integration       | Claude via `langchain-anthropic`, handling API constraints (Opus 4.8)  |
| Best practices        | Config/ingest/chain/UI separation, secrets out of code, cited sources  |
| Productionization     | Dockerized, public deployment, cost/abuse guardrails, inspectable UI   |

---

## Resources

- [LangChain documentation](https://docs.langchain.com/oss/python/langchain/overview)
- [LangChain RAG guide](https://docs.langchain.com/oss/python/langchain/rag)
- [Anthropic — Console & docs](https://console.anthropic.com/)
- [Voyage AI — embeddings](https://www.voyageai.com/)
- [RAGAS — RAG evaluation](https://docs.ragas.io/)
