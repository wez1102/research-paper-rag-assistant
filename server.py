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

Operational bits (see docs/DEPLOYMENT.md):
  /healthz  - liveness: the process is up. Never touches the index or the LLM.
  /readyz   - readiness: 200 only when the index and models are loaded; also says
              whether an LLM is reachable (without one, answers fall back to quotes).
  Every response carries an X-Request-ID, and each /api/ask writes one JSON log line
  with its stage timings, so a slow or failed answer can be traced afterwards.
  API_KEYS=... turns on bearer-token auth for /api/*.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from config import settings
from generate import LLMError, answer_events, model_for, ollama_models, resolve_provider
from retrieve import MODES, Retriever

STATIC = Path(__file__).resolve().parent / "static"

# One JSON object per line, so a log platform (Cloud Logging, Loki, `jq`) can filter
# and chart on the fields without regexes.
log = logging.getLogger("rag.server")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False


def log_event(**fields) -> None:
    log.info(json.dumps({"ts": round(time.time(), 3), **fields}, default=str))


# Loaded once at startup (embedding model, BM25, cross-encoder: ~10 s on a laptop).
# If loading fails, the server still starts, /readyz explains why, and /api/ask
# answers 503 - much easier to debug than a container stuck in a crash loop.
state: dict = {"retriever": None, "load_error": None, "started": time.time()}
# The embedding model and cross-encoder aren't guaranteed to be thread-safe, and
# FastAPI runs streaming generators in a thread pool. Retrieval is fast enough
# (~1-2 s) that serialising it is simpler than worrying about it.
retrieval_lock = threading.Lock()
# Generation slots. Ollama answers one request at a time on a laptop anyway, so
# capping concurrency here keeps waiting requests visible (and time-limited) instead
# of silently stuck inside Ollama's own queue.
_gen_slots: asyncio.Semaphore | None = None


def gen_slots() -> asyncio.Semaphore:
    global _gen_slots
    if _gen_slots is None:
        _gen_slots = asyncio.Semaphore(settings.max_concurrent_generations)
    return _gen_slots


def load_retriever() -> None:
    try:
        state["retriever"] = Retriever(load_reranker=True)
        state["load_error"] = None
        log_event(event="startup", status="ready", n_chunks=len(state["retriever"].ids))
    except Exception as err:                      # keep serving /healthz and /readyz
        state["retriever"] = None
        state["load_error"] = f"{type(err).__name__}: {err}"
        log_event(event="startup", status="not_ready", error=state["load_error"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(load_retriever)
    yield


app = FastAPI(title="Research Paper RAG Assistant", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def request_id(request: Request, call_next):
    """Give every request an id (or keep the caller's), echo it back, log the request."""
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    request.state.request_id = rid
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    if request.url.path.startswith("/api/"):
        log_event(event="http", request_id=rid, method=request.method, path=request.url.path,
                  status=response.status_code, ms=round((time.perf_counter() - t0) * 1000, 1))
    return response


def get_retriever() -> Retriever:
    """Split out so the tests can swap in a fake retriever without a real index."""
    r = state["retriever"]
    if r is None:
        raise HTTPException(503, f"The search index isn't loaded: {state['load_error'] or 'still starting'}")
    return r


def require_key(request: Request) -> None:
    """Bearer-token auth, on only when API_KEYS is set. Constant-time comparison."""
    keys = settings.api_keys
    if not keys:
        return
    header = request.headers.get("authorization", "")
    token = header[7:] if header.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
    if not any(hmac.compare_digest(token.encode(), k.encode()) for k in keys):
        raise HTTPException(401, "Missing or wrong API key", headers={"WWW-Authenticate": "Bearer"})


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
    except (LLMError, OSError, ValueError):
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


@app.get("/healthz")
def healthz() -> dict:
    """Liveness: answers as long as the process is alive. Cheap on purpose."""
    return {"status": "ok", "uptime_s": round(time.time() - state["started"], 1)}


@app.get("/readyz")
def readyz() -> JSONResponse:
    """
    Readiness, checked dependency by dependency. Not ready (503) only if retrieval
    can't work. A missing LLM makes us "degraded", not unready, because the no-LLM
    quoting mode still gives grounded answers.
    """
    r = state["retriever"]
    checks: dict = {}
    if r is None:
        checks["index"] = {"ok": False, "error": state["load_error"] or "still starting"}
    else:
        checks["index"] = {"ok": True, "collection": getattr(r, "collection_name", ""), "n_chunks": len(r.ids),
                           "manifest": (getattr(r, "manifest", {}) or {}).get("manifest_hash", "")[:12]}
        checks["reranker"] = {"ok": getattr(r, "_reranker", True) is not None}
    provider = resolve_provider()
    if provider == "ollama" or settings.llm_provider in ("auto", "ollama"):
        try:
            names = [m["name"] for m in ollama_models()]
            checks["ollama"] = {"ok": bool(names), "models": len(names)}
        except Exception as err:
            checks["ollama"] = {"ok": False, "error": str(err)[:200]}
    checks["llm"] = {"ok": provider != "extractive" and checks.get("ollama", {}).get("ok", provider != "ollama"),
                     "provider": provider}
    ready = checks["index"]["ok"] and checks.get("reranker", {}).get("ok", False)
    status = "ready" if ready and checks["llm"]["ok"] else ("degraded" if ready else "not_ready")
    return JSONResponse({"status": status, "checks": checks}, status_code=200 if ready else 503)


@app.get("/api/models", dependencies=[Depends(require_key)])
def models() -> dict:
    providers = list_providers()
    return {"providers": providers, "default": default_choice(providers)}


@app.get("/api/info", dependencies=[Depends(require_key)])
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


@app.get("/api/benchmark", dependencies=[Depends(require_key)])
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


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.post("/api/ask", dependencies=[Depends(require_key)])
async def ask(req: AskRequest, request: Request) -> StreamingResponse:
    rid = request.state.request_id
    retriever = get_retriever()                         # 503 straight away if the index isn't loaded
    provider, model = await run_in_threadpool(_check, req)

    # Wait (a bounded time) for a generation slot. Better a clear 503 + Retry-After
    # than a request that hangs for minutes behind others.
    slots = gen_slots()
    try:
        await asyncio.wait_for(slots.acquire(), timeout=settings.queue_timeout)
    except asyncio.TimeoutError:
        log_event(event="ask", request_id=rid, outcome="busy")
        raise HTTPException(503, "The server is busy answering other questions. Try again shortly.",
                            headers={"Retry-After": "10"})

    def start():
        stream = answer_events(req.question.strip(), retriever, mode=req.mode, k=req.k,
                               paper_id=req.paper_id, provider=provider, model=model)
        with retrieval_lock:
            first = next(stream)                     # retrieval happens here, under the lock
        return stream, first

    async def events():
        # Everything the model does runs in worker threads; between pieces we check
        # whether the browser is still there. If it left, we close the generator,
        # which closes the HTTP stream to Ollama, which stops the generation - so an
        # abandoned question doesn't keep the GPU/CPU busy.
        t0 = time.perf_counter()
        info = {"event": "ask", "request_id": rid, "provider": provider, "model": model, "mode": req.mode,
                "scope": "paper" if req.paper_id else "corpus", "k": req.k}
        stream = None
        ttft = None
        try:
            stream, first = await run_in_threadpool(start)
            first["request_id"] = rid
            info.update(retrieval_s=round(first.get("retrieval_s", 0), 4),
                        **{k: round(v, 4) for k, v in (first.get("timings") or {}).items()})
            yield _sse(first)
            while True:
                event = await run_in_threadpool(next, stream, None)
                if event is None:
                    break
                if await request.is_disconnected():
                    info["outcome"] = "client_disconnected"
                    return
                if event["type"] == "token" and ttft is None:
                    ttft = time.perf_counter() - t0
                if event["type"] == "final":
                    ans = event["answer"]
                    ans["request_id"] = rid
                    info.update(outcome="refused" if ans["refused"] else "answered", reason=ans.get("reason", ""),
                                generation_s=round(ans.get("generation_s", 0), 3), cite_repair=ans.get("cite_repair"))
                yield _sse(event)
        except LLMError as err:
            info.update(outcome="error", error_code=err.code, error=str(err)[:300])
            yield _sse({"type": "error", "code": err.code, "message": str(err), "request_id": rid})
        except Exception as err:                    # surface anything else in the UI instead of a dead stream
            info.update(outcome="error", error_code="internal", error=f"{type(err).__name__}: {err}"[:300])
            yield _sse({"type": "error", "code": "internal", "message": f"{type(err).__name__}: {err}",
                        "request_id": rid})
        finally:
            if stream is not None:
                try:
                    stream.close()
                except ValueError:                  # still running in its thread; it ends on its own
                    pass
            slots.release()
            info.setdefault("outcome", "client_disconnected")
            info.update(ttft_s=round(ttft, 3) if ttft is not None else None,
                        total_s=round(time.perf_counter() - t0, 3))
            log_event(**info)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Request-ID": rid})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8000")))
