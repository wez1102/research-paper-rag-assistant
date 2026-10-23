# Container for Hugging Face Spaces (Docker SDK) - also works anywhere Docker runs.
#   docker build -t paper-rag . && docker run -p 7860:7860 -e ANTHROPIC_API_KEY=... paper-rag
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only PyTorch first: the default wheel drags in ~3 GB of CUDA libraries we don't need.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the index while the image is built, so the Space starts instantly and
# every container serves exactly the same index that was evaluated.
RUN python ingest.py

# Spaces runs containers as uid 1000.
RUN useradd -m -u 1000 user && chown -R user /app
USER user

EXPOSE 7860
# The web UI (server.py). The older Streamlit page still works too:
#   CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
