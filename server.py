"""
The web app: a small JSON/streaming API plus the single-page UI in static/.

    uvicorn server:app --reload          # then open http://localhost:8000

Why not just Streamlit (app.py still works)? Two things a reader of the code
should be able to see working:
  * the model picker lists whatever is actually installed in your local Ollama,
    plus any API provider you've configured, and you can switch per question;
  * answers stream in token by token, and the sources show up the moment
    retrieval finishes - before the model has written a word.

The API is deliberately thin. Every answer goes through generate.answer_events(),
which uses the same retrieval, refusal and citation-repair code that evaluate.py
scores, so the demo and the benchmark can't drift apart.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import settings
from generate import answer_events, model_for, ollama_models, resolve_provider
from retrieve import MODES, Retriever

STATIC = Path(__file__).resolve().parent / "static"

# Loaded once at startup (embedding model, BM25, cross-encoder: ~10 s on a laptop).
state: dict = {"retriever": None}
# The embedding model and cross-encoder aren't guaranteed to be thread-safe, and
# FastAPI runs streaming generators in a thread pool. Retrieval is fast enough
# (~1-2 s) that serialising it is simpler than worrying about it.
retrieval_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["retriever"] = Retriever(load_reranker=True)
    yield


app = FastAPI(title="Research Paper RAG Assistant", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def get_retriever() -> Retriever:
    """Split out so the tests can swap in a fake retriever without a real index."""
    return state["retriever"]


# ---------------------------------------------------------------------------
# What can answer questions right now?
# ---------------------------------------------------------------------------

def list_providers() -> list[dict]:
    """
    Everything the model picker can offer, with a plain-language note when an
    option isn't usable yet. Local Ollama models are discovered live; API
    providers appear only when their key is set (the key itself never leaves
    the server).
    """
    providers = []

    try:
        models = ollama_models()
        note = "" if models else "Ollama is running but has no chat models yet. Try: ollama pull qwen3:8b"
        providers.append({"id": "ollama", "label": "On this computer (Ollama)", "available": bool(models),
                          "models": models, "note": note, "host": settings.ollama_host})
    except (urllib.error.URLError, OSError, ValueError):
        providers.append({"id": "ollama", "label": "On this computer (Ollama)", "available": False, "models": [],
                          "note": f"Ollama isn't running at {settings.ollama_host}. Start it with: ollama serve",
                          "host": settings.ollama_host})

    if os.getenv("OPENAI_API_KEY"):
        host = urlparse(settings.openai_base_url).hostname if settings.openai_base_url else "api.openai.com"
        providers.append({"id": "openai", "label": f"API ({host})", "available": True,
                          "models": [{"name": model_for("openai")}], "note": ""})
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append({"id": "anthropic", "label": "API (Anthropic)", "available": True,
                          "models": [{"name": model_for("anthropic")}], "note": ""})

    providers.append({"id": "extractive", "label": "No LLM", "available": True,
                      "models": [{"name": "quote-sentences"}],
                      "note": "Quotes the most relevant sentences instead of writing an answer."})
    return providers


def default_choice(providers: list[dict]) -> dict:
    """Same preference order as the CLI (resolve_provider), but only among usable options."""
    usable = {p["id"]: p for p in providers if p["available"]}
    wanted = resolve_provider()
    p = usable.get(wanted) or usable.get("ollama") or next(iter(usable.values()))
    names = [m["name"] for m in p["models"]]
    preferred = model_for(p["id"])
    return {"provider": p["id"], "model": preferred if preferred in names else names[0]}


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/models")
def models() -> dict:
    providers = list_providers()
    return {"providers": providers, "default": default_choice(providers)}


@app.get("/api/info")
def info() -> dict:
    r = get_retriever()
    return {
        "papers": [{"id": pid, "title": title} for pid, title in r.papers()],
        "n_chunks": len(r.ids),
        "modes": [{"id": "dense", "label": "Dense"}, {"id": "hybrid", "label": "Hybrid"},
                  {"id": "hybrid_rerank", "label": "Hybrid + rerank"}],
        "default_mode": "hybrid_rerank",
        "top_k": settings.top_k,
    }


@app.get("/api/benchmark")
def benchmark() -> dict:
    """The evaluation numbers, read straight from results/ so the UI never shows stale figures."""
    out: dict = {"retrieval": [], "answers": []}
    path = settings.results_dir / "results.csv"
    if path.exists():
        out["retrieval"] = pd.read_csv(path).fillna("").to_dict(orient="records")
    for f in sorted(settings.results_dir.glob("answers_summary_*.csv")):
        out["answers"].extend(pd.read_csv(f).fillna("").to_dict(orient="records"))
    return out


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    provider: str
    model: str = ""
    mode: str = "hybrid_rerank"
    k: int = Field(default=5, ge=1, le=10)
    paper_id: str | None = None


def _check(req: AskRequest) -> tuple[str, str]:
    """Validate the request against what's really available. Returns (provider, model)."""
    if req.mode not in MODES:
        raise HTTPException(400, f"Unknown retrieval mode '{req.mode}'")
    if req.paper_id and req.paper_id not in {pid for pid, _ in get_retriever().papers()}:
        raise HTTPException(400, "That paper isn't in the index")
    provider = next((p for p in list_providers() if p["id"] == req.provider), None)
    if provider is None or not provider["available"]:
        raise HTTPException(400, provider["note"] if provider else f"Unknown model source '{req.provider}'")
    names = [m["name"] for m in provider["models"]]
    if req.provider == "ollama":
        if req.model not in names:
            raise HTTPException(400, f"'{req.model}' isn't installed in Ollama. Installed: {', '.join(names)}")
        return "ollama", req.model
    # For paid APIs the server decides the model, so a public demo can't be pointed at an expensive one.
    return req.provider, names[0]


@app.post("/api/ask")
def ask(req: AskRequest) -> StreamingResponse:
    provider, model = _check(req)

    def events():
        # Server-Sent Events: one "data: {...}" line per event, flushed as it happens.
        try:
            with retrieval_lock:
                stream = answer_events(req.question.strip(), get_retriever(), mode=req.mode, k=req.k,
                                       paper_id=req.paper_id, provider=provider, model=model)
                first = next(stream)               # retrieval happens here, under the lock
            yield f"data: {json.dumps(first)}\n\n"
            for event in stream:
                yield f"data: {json.dumps(event)}\n\n"
        except urllib.error.URLError:
            msg = f"Couldn't reach Ollama at {settings.ollama_host}. Is it still running?"
            yield f"data: {json.dumps({'type': 'error', 'message': msg})}\n\n"
        except Exception as err:                    # surface API errors in the UI instead of a dead stream
            yield f"data: {json.dumps({'type': 'error', 'message': f'{type(err).__name__}: {err}'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8000")))
