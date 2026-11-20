# Research Paper RAG Assistant - CPU image.
#   docker build -t paper-rag .
#   docker run -p 7860:7860 -e API_KEYS=change-me paper-rag
# The image contains the models, the dataset file and the built index, so it starts
# without network access and every container serves exactly the evaluated index.
# An LLM is optional: point LLM_PROVIDER/OLLAMA_HOST/OPENAI_* at one (see .env.example),
# otherwise answers fall back to quoting the best sentences.
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only PyTorch first: the default wheel drags in ~3 GB of CUDA libraries we don't need.
COPY requirements.txt .
RUN pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# Models + dataset in their own layer, so code changes don't re-download them.
COPY scripts/download_assets.py scripts/
RUN python scripts/download_assets.py /app/assets
ENV QASPER_PARQUET=/app/assets/qasper/validation.parquet \
    EMBED_MODEL=/app/assets/models/bge-small-en-v1.5 \
    RERANK_MODEL=/app/assets/models/ms-marco-MiniLM-L-6-v2

COPY . .
# Build the index at image build time (idempotent: the manifest check makes re-runs no-ops).
RUN python ingest.py

# Nothing at runtime needs the Hub.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Run as an unprivileged user (Hugging Face Spaces also expects uid 1000).
RUN useradd -m -u 1000 user && chown -R user /app
USER user

EXPOSE 7860
# Liveness only; orchestrators should use /readyz for readiness (it waits for the models).
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/healthz', timeout=3).status == 200 else 1)"

# One worker: the models live in process memory (~600 MB). Scale with more containers.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--timeout-graceful-shutdown", "20"]
