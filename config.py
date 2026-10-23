"""
Central settings for the project.

Everything that you might want to tweak (model names, chunk sizes, paths, which
LLM to call) lives here so the other modules don't hard-code anything. Values
can be overridden with environment variables or a local .env file, which is
handy when deploying (e.g. on Hugging Face Spaces you set them as secrets).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env if it exists. It's git-ignored, so API keys never end up in the repo.
load_dotenv()

ROOT = Path(__file__).resolve().parent


def _env(name: str, default: str) -> str:
    """Small helper so the dataclass below stays readable."""
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    # --- Data -----------------------------------------------------------------
    # QASPER on the Hub is an old script-based dataset, which recent versions of
    # `datasets` refuse to run. The Hub also publishes an auto-converted Parquet
    # copy on the `refs/convert/parquet` branch, so we read that file directly.
    qasper_repo: str = "allenai/qasper"
    qasper_revision: str = "refs/convert/parquet"
    qasper_file: str = "qasper/validation/0000.parquet"
    # Point this at a local parquet file if you already downloaded it (optional).
    qasper_local_path: str = _env("QASPER_PARQUET", "")
    num_papers: int = int(_env("NUM_PAPERS", "50"))

    data_dir: Path = ROOT / "data"
    # Some network/shared drives don't support SQLite file locking, which Chroma
    # needs. CHROMA_DIR lets you put the index on a local disk in that case.
    chroma_dir: Path = Path(_env("CHROMA_DIR", str(ROOT / "data" / "chroma")))
    results_dir: Path = ROOT / "results"

    # --- Chunking -------------------------------------------------------------
    # Counted in tokens of the embedding model's own tokenizer, so a 450-token
    # chunk really fits inside bge-small's 512-token window.
    chunk_size: int = int(_env("CHUNK_SIZE", "450"))
    chunk_overlap: int = int(_env("CHUNK_OVERLAP", "50"))

    # --- Models ---------------------------------------------------------------
    # These can be Hub ids or local folders (useful offline).
    embed_model: str = _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    rerank_model: str = _env("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    # bge models were trained with this prefix on the *query* side only.
    query_instruction: str = "Represent this sentence for searching relevant passages: "

    # --- Retrieval ------------------------------------------------------------
    top_k: int = 5
    rerank_candidates: int = 30   # how many hybrid results the cross-encoder re-scores
    # How a chunk's cross-encoder score is built: "chunk" (whole chunk), "maxp" (best
    # paragraph), "max" (better of the two) or "mean". Chosen on the dev set.
    rerank_how: str = _env("RERANK_HOW", "max")
    rerank_prior: float = float(_env("RERANK_PRIOR", "0.5"))  # weight of the first-stage order after reranking
    fusion_pool: int = 50         # how deep each retriever goes before RRF
    rrf_k: int = 60               # the standard constant from the original RRF paper

    # --- Generation -----------------------------------------------------------
    # auto | ollama | openai | anthropic | extractive
    llm_provider: str = _env("LLM_PROVIDER", "auto")
    llm_model: str = _env("LLM_MODEL", "")
    ollama_host: str = _env("OLLAMA_HOST", "http://localhost:11434")
    openai_base_url: str = _env("OPENAI_BASE_URL", "")  # e.g. Groq's OpenAI-compatible URL
    max_answer_tokens: int = 400
    # Which system prompt generate.py uses (see PROMPTS there). v2 = answer-first.
    prompt_version: str = _env("PROMPT_VERSION", "v2")
    # If the model answers without a single valid [n], ask it once to add them (see
    # REPAIR_PROMPT in generate.py) instead of refusing straight away. CITE_REPAIR=0 turns it off.
    cite_repair: bool = _env("CITE_REPAIR", "1") == "1"
    # If the best cross-encoder score is below this, we don't even ask the LLM - the
    # question is clearly about something else. The ms-marco cross-encoder outputs raw
    # logits. On the dev papers (50-99): off-topic questions ("What is the capital of
    # France?", "Who wrote Hamlet?"...) score -11 to -7.7, but so do some perfectly good
    # yes/no questions ("Do they report results only on English?" got -8.8 while its
    # top chunk held the answer) - MS MARCO-trained rerankers are bad at yes/no.
    # So this is only a floor for the obvious cases; anything in the grey zone above it
    # goes to the LLM, whose prompt tells it to refuse when the sources don't help.
    min_rerank_score: float = float(_env("MIN_RERANK_SCORE", "-9.0"))

    # "auto" picks CUDA, then Apple's MPS, then CPU. Set DEVICE=cpu to force CPU.
    device: str = _env("DEVICE", "auto")

    seed: int = 42

    # Default model names per provider, used when LLM_MODEL isn't set.
    default_models: dict = field(default_factory=lambda: {
        "ollama": "qwen3:8b",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-haiku-5-5",
    })

    def collection_name(self, chunk_size: int | None = None) -> str:
        """One Chroma collection per chunk size, so experiments don't overwrite each other."""
        return f"qasper_chunks_{chunk_size or self.chunk_size}"


def _resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


settings = Settings()
# frozen dataclass, so resolve the device once with object.__setattr__
object.__setattr__(settings, "device", _resolve_device(settings.device))
