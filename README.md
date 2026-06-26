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

Deployment is automated via GitHub Actions (`.github/workflows/deploy.yml`): every push runs the
tests, and a push to `dev` or `main` mirrors the repo to a Hugging Face Space, which rebuilds the
Docker image. (The deploy step is skipped until `HF_SPACE` is set, so it never fails before setup —
and you can validate the live deploy from `dev` before merging to `main`.) One-time setup:

1. **Create the Space** — on Hugging Face, *New Space* → **SDK: Docker** (the `Dockerfile` and this
   README's frontmatter are all it needs).
2. **Add the runtime secret** — in the Space's *Settings → Variables and secrets*, add
   `ANTHROPIC_API_KEY` (injected into the container at runtime).
3. **Wire up auto-deploy** — in the GitHub repo's *Settings → Secrets and variables → Actions*:
   - secret `HF_TOKEN` — a Hugging Face access token with **write** scope;
   - variables `HF_USERNAME` (your HF username) and `HF_SPACE` (the Space name).
4. **Ship it** — push to `dev` (or `main`). The workflow pushes to the Space; the build runs
   ingestion, bakes the index in, and serves the app on port 8501 at your public Space URL.

> Manual alternative (no CI): add the Space as a git remote and `git push` to it, then set the
> `ANTHROPIC_API_KEY` secret in the Space.

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
