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
    # Paper slices of the QASPER validation split, as (first row, number of papers).
    # They never share a paper, so tuning on one can't leak into another:
    #   app   - the 50 papers the web app serves. Also the "test" set of the first
    #           round (Oct 2026). Several choices were looked at on it back then
    #           (chunk size, for one), so it's no longer a clean held-out set.
    #   dev   - every setting is chosen here.
    #   final - opened once, after all choices are frozen, to report numbers.
    splits: dict = field(default_factory=lambda: {"app": (0, 50), "dev": (50, 50), "final": (100, 50)})

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
    rerank_candidates: int = int(_env("RERANK_CANDIDATES", "30"))  # how many hybrid results the cross-encoder re-scores
    # How a chunk's cross-encoder score is built: "chunk" (whole chunk), "maxp" (best
    # paragraph), "max" (better of the two) or "mean". Round 1 picked "max" on dev.
    # Round 2 switched to "maxp": it skips the long whole-chunk pairs, which halves
    # rerank time on CPU (p50 2.5 s -> 1.1 s per query, results/v2/bench/), for one
    # question less in Hit@5 on dev (n=109) and the same MRR. RERANK_HOW=max restores it.
    rerank_how: str = _env("RERANK_HOW", "maxp")
    rerank_prior: float = float(_env("RERANK_PRIOR", "0.5"))  # weight of the first-stage order after reranking
    fusion_pool: int = 50         # how deep each retriever goes before RRF
    rrf_k: int = 60               # the standard constant from the original RRF paper
    # Weights of the dense and BM25 lists inside RRF. 1,1 is plain RRF.
    rrf_weights: tuple = tuple(float(x) for x in _env("RRF_WEIGHTS", "1,1").split(","))
    # Corpus-wide search only (no paper picked): how much a chunk is helped by the
    # rest of its paper also matching the question. 0 turns it off. See
    # Retriever.paper_aware() and docs/EVALUATION.md for how it was chosen.
    paper_boost: float = float(_env("PAPER_BOOST", "0"))
    # How papers are ranked for that boost: "sum" (add up the paper's chunk scores)
    # or "abstract" (match the question against each paper's title + abstract only).
    paper_prior: str = _env("PAPER_PRIOR", "abstract")
    # Corpus-wide questions that don't say which paper they mean ("What was the
    # baseline?") get a "which paper?" reply with likely candidates instead of an
    # answer that is usually about the wrong paper. A question counts as specific when
    # its rarest word appears in at most this share of the papers (0.06 = 3 of 50,
    # picked on the dev split). CLARIFY_GENERIC=0 turns it off.
    clarify_generic: bool = _env("CLARIFY_GENERIC", "1") == "1"
    generic_paper_share: float = float(_env("GENERIC_PAPER_SHARE", "0.06"))
    # Longest input the cross-encoder reads, in tokens. Longer = slower.
    rerank_max_length: int = int(_env("RERANK_MAX_LENGTH", "512"))

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

    # --- Serving / reliability ----------------------------------------------
    # Seconds to wait for the LLM to connect, and for each next piece of a streamed
    # answer. A model that is loading into memory can take a while to say anything.
    llm_connect_timeout: float = float(_env("LLM_CONNECT_TIMEOUT", "5"))
    llm_read_timeout: float = float(_env("LLM_READ_TIMEOUT", "120"))
    # Hard cap on one whole answer, so a stuck model can't hold a request forever.
    llm_total_timeout: float = float(_env("LLM_TOTAL_TIMEOUT", "180"))
    # How many answers may be generated at the same time. Ollama runs them one by
    # one anyway, so extra requests wait here instead of piling up inside Ollama.
    max_concurrent_generations: int = int(_env("MAX_CONCURRENT_GENERATIONS", "2"))
    # How long a request waits for a free generation slot before getting a 503.
    queue_timeout: float = float(_env("QUEUE_TIMEOUT", "30"))
    # Comma-separated keys. When set, every /api/* call except the health checks
    # needs "Authorization: Bearer <key>". Empty = open (fine on your own laptop).
    api_keys: tuple = tuple(k.strip() for k in _env("API_KEYS", "").split(",") if k.strip())

    # "auto" picks CUDA, then Apple's MPS, then CPU. Set DEVICE=cpu to force CPU.
    device: str = _env("DEVICE", "auto")

    seed: int = 42

    # Default model names per provider, used when LLM_MODEL isn't set.
    default_models: dict = field(default_factory=lambda: {
        "ollama": "qwen3:8b",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-haiku-5-5",
    })

    def collection_name(self, chunk_size: int | None = None, split: str = "app") -> str:
        """One Chroma collection per split and chunk size, so experiments don't overwrite each other."""
        size = chunk_size or self.chunk_size
        # The app split keeps its original name so existing indexes keep working.
        return f"qasper_chunks_{size}" if split == "app" else f"qasper_{split}_chunks_{size}"

    def questions_path(self, split: str = "app") -> Path:
        return self.data_dir / ("questions.jsonl" if split == "app" else f"questions_{split}.jsonl")

    def unanswerable_path(self, split: str = "app") -> Path:
        """QASPER questions every annotator marked unanswerable (used for refusal checks)."""
        return self.data_dir / f"unanswerable_{split}.jsonl"


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
