# Implementation roadmap

Progress toward a public RAG demo — minimal but production-ready.
Each step has a verifiable **validation criterion**.

Legend: `[ ]` to do · `[x]` done.

---

## Step 0 — Foundations  `[x]`

- [x] `.gitignore`, `.env.example`, `requirements.txt`, `README.md`, `objective.md`

**Valid if:** `pip install -r requirements.txt` succeeds in a clean venv.

---

## Step 1 — Configuration  `src/config.py`  `[x]`

- [x] Load `.env`, `get_llm()` (no `temperature` — see Opus 4.8 gotcha), `get_embeddings()`
- [x] Guardrails (`MAX_TOKENS`, `MAX_QUESTIONS_PER_SESSION`), pricing table

**Valid if:** `python -c "import src.config"` runs without error.

---

## Step 2 — Ingestion  `src/ingest.py`  `[x]`

- [x] Fetch markdown `.md` endpoints (JS-rendered site) → `RecursiveCharacterTextSplitter` (1000/150) → ChromaDB (cosine)
- [x] Per-page diagnostic to confirm each URL produced text

**Valid if:** `python -m src.ingest` creates `vectorstore/` and a search returns relevant chunks.

---

## Step 3 — RAG chain  `src/rag.py`  `[x]`

- [x] `retrieve` (with scores), `build_context`, `generate` — decomposed for visibility
- [x] Prompt constrained to the context, sourcing, refusal when info is missing

**Valid if:** a covered question → correct, sourced answer; an off-topic one → explicit refusal.

---

## Step 4 — UI  `src/app.py`  `[x]`

- [x] Streamlit chat + history
- [x] **Everything inspectable**: sources + scores, context sent, timings, tokens, cost
- [x] Guardrails: Haiku, `max_tokens`, per-session limit, optional own API key

**Valid if:** `PYTHONPATH=. streamlit run src/app.py` allows a conversation and shows the internals.

---

## Step 5 — Containerization & public deployment  `[x]`

- [x] `Dockerfile` (HF Spaces, port 8501) — index built at build time and baked into the image
- [x] `.dockerignore`, HF Spaces frontmatter in `README.md`
- [x] `ANTHROPIC_API_KEY` as a host secret (never in the repo)

**Valid if:** `docker build` succeeds; the container serves the app; the Space responds publicly.

---

## Step 6 — Quality  `[~]`

- [x] `tests/` : smoke tests with no network and no key
- [ ] Retrieval integration test (needs a built index)
- [ ] CI (GitHub Actions: lint + tests + Docker build)

**Valid if:** `pytest` passes and CI is green.

---

## Possible improvements (out of scope)

- Hybrid semantic + BM25 search (`EnsembleRetriever`)
- RAG evaluation (faithfulness / answer relevancy) with RAGAS
- Conversational memory (rewrite the question from history)
- Streaming the Claude answer in the UI
- Global rate-limiting (beyond per-session) if traffic warrants it
