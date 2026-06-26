# Demo image, ready for Hugging Face Spaces (Docker SDK, port 8501).
# The vector index is built at build time (local embeddings, no key required)
# and baked into the image -> fast, deterministic startup.
FROM python:3.12-slim

# HF Spaces runs the container as user uid 1000.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    PERSIST_DIR=/home/user/app/vectorstore \
    USER_AGENT=demo-rag/0.1

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user src/ src/

# Build and bake the index into the image.
RUN python -m src.ingest

# Make the `src` package importable at runtime: the Streamlit entrypoint only
# adds src/ to PYTHONPATH, not the project root.
ENV PYTHONPATH=/home/user/app

EXPOSE 8501
# fileWatcherType=none: no hot-reload in a container — stops the Streamlit
# watcher from crawling `transformers` submodules (the torchvision noise).
CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.fileWatcherType=none"]
