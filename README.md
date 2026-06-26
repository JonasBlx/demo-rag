---
title: RAG LangChain Docs
emoji: 🔗
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
---

# RAG over the LangChain documentation

A **public** demo of a minimal but production-ready **RAG** (Retrieval-Augmented Generation)
architecture: ask the LangChain documentation questions in natural language, with **Claude**
(Anthropic) for generation, **source citations**, and the **whole pipeline inspectable**
(retrieved chunks, scores, the context sent to the model, timings, tokens, cost).

> The YAML block at the top of this file configures the [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker) deployment (Docker SDK, port 8501).

---

## Live demo

Deployed on Hugging Face Spaces (Docker). Reachable by anyone, no installation. Questions are
expected in **English** (the docs and the embedding model are English).

Guardrails (owner-paid demo, open to all):

- **`claude-haiku-4-5`** model (fast and cheap),
- capped `max_tokens`,
- per-session question limit,
- visitors can **use their own API key**.

## Architecture

```
INGESTION   LangChain docs (web) → chunking → embedding → ChromaDB   (at image build time)
                                                              │
RETRIEVAL   question → embedding → cosine similarity → top-k chunks
                                                              │
GENERATION  chunks + question → prompt → Claude → answer + sources
```

## Tech stack

| Component     | Default choice                            | Alternative                       |
|---------------|-------------------------------------------|-----------------------------------|
| Orchestration | LangChain (LCEL)                          | LlamaIndex                        |
| LLM           | **Claude Haiku 4.5** (`claude-haiku-4-5`) | `claude-opus-4-8` (quality)       |
| Embeddings    | **sentence-transformers** (local)         | Voyage AI (hosted)                |
| Vector store  | ChromaDB (persisted, baked into image)    | FAISS, pgvector                   |
| UI            | Streamlit                                 | Gradio                            |
| Hosting       | Hugging Face Spaces (Docker)              | Render / Fly.io / Cloud Run       |

> **Note:** Anthropic has **no** embeddings API — Claude generates, a separate *embedder* (local
> or Voyage AI) indexes/searches. Rationale in [`objective.md`](objective.md).

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                # fill in ANTHROPIC_API_KEY

python -m src.ingest                                # 1. build the index (once)
PYTHONPATH=. streamlit run src/app.py               # 2. start the chatbot → http://localhost:8501
```

## Run with Docker (same as prod)

```bash
docker build -t demo-rag .                          # the index is built during the build
docker run -p 8501:8501 -e ANTHROPIC_API_KEY=sk-... demo-rag
```

## Deploy to Hugging Face Spaces

1. Create a Space → **SDK: Docker** (the `Dockerfile` and this README's frontmatter are enough).
2. Push this repo to the Space (or connect it to GitHub).
3. In **Settings → Variables and secrets**, add the `ANTHROPIC_API_KEY` secret.
4. The build runs ingestion and bakes the index in; the app starts on port 8501.

## Configuration (environment variables)

See [`.env.example`](.env.example). Main ones: `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`,
`MAX_TOKENS`, `RETRIEVAL_K`, `EMBEDDING_BACKEND`, `MAX_QUESTIONS_PER_SESSION`.

## Project layout

```
demo-rag/
├── src/
│   ├── config.py      # 12-factor config: LLM + embeddings + guardrails
│   ├── ingest.py      # ingestion → indexing (run at build time)
│   ├── rag.py         # RAG chain, decomposed (retrieve / context / generate)
│   └── app.py         # Streamlit UI, fully inspectable
├── tests/             # smoke tests (no network, no key)
├── docs/ROADMAP.md    # plan + validation criteria
├── objective.md       # refined, justified specification
├── Dockerfile         # prod image (= HF Spaces demo)
├── .dockerignore
├── .env.example
└── requirements.txt
```

## Design decisions

- **Haiku by default + caps**: the demo is open to all and owner-paid, so we bound the cost
  (cheap model, `max_tokens`, per-session limit, optional bring-your-own-key).
- **No `temperature`** passed to `ChatAnthropic`: Opus 4.7/4.8 reject it (400 error); omitting it
  gives a single code path valid for every Claude model.
- **English embedder** (`all-MiniLM-L6-v2`): questions are assumed English, matching the docs.
  Switchable to Voyage AI via `EMBEDDING_BACKEND`.
- **Index baked into the image** (built at build time): fast, deterministic startup, stateless
  container.
- **Everything inspectable**: sources + scores, the context sent to the model, timings, tokens,
  cost.

## Development

```bash
pip install pytest ruff
pytest        # smoke tests
ruff check .  # lint
```

## Documentation

- [`objective.md`](objective.md) — objectives, justified stack, guardrails, reference code
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — steps and validation criteria
