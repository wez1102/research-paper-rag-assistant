"""
Tests for the web API (server.py). They use a fake retriever and a fake LLM, so
they run in a second with no index, no models and no Ollama.

    pytest -q tests/test_server.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generate  # noqa: E402
import server  # noqa: E402
from retrieve import RetrievedChunk  # noqa: E402


class FakeRetriever:
    """Two papers, a handful of chunks. Off-topic questions get hopeless rerank scores."""

    ids = ["p1-000", "p1-001", "p2-000"]

    def papers(self):
        return [("p1", "Learning Autocomplete Systems as a Communication Game"), ("p2", "Another Paper")]

    def retrieve(self, question, mode="hybrid_rerank", k=None, paper_id=None, timings=None):
        score = -11.0 if "France" in question else 4.2
        chunks = [
            RetrievedChunk("p1-001", "p1", self.papers()[0][1], "Experiments > User study",
                           "We recruited 100 crowdworkers on Amazon Mechanical Turk.", [3], score, "rerank"),
            RetrievedChunk("p1-000", "p1", self.papers()[0][1], "Abstract",
                           "We study textual autocomplete as a communication game.", [0], score - 3, "rerank"),
        ]
        return chunks[: k or 5]


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient

    server.state["retriever"] = FakeRetriever()
    server.state["load_error"] = None
    server._gen_slots = None              # fresh semaphore for this test's event loop
    monkeypatch.setattr(server, "ollama_models", lambda: [
        {"name": "qwen3:8b", "size_gb": 5.2, "params": "8.2B", "quant": "Q4_K_M"}])
    monkeypatch.setattr(server, "resolve_provider", lambda: "ollama")
    monkeypatch.setattr(server, "model_for", lambda p: "qwen3:8b" if p == "ollama" else "quote-sentences")
    # Without `with`, TestClient skips the lifespan, so the real index is never loaded.
    return TestClient(server.app)


def events(resp) -> list[dict]:
    return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]


def test_models_lists_local_ollama_models_and_fallback(client):
    data = client.get("/api/models").json()
    ids = [p["id"] for p in data["providers"]]
    assert ids[0] == "ollama" and "extractive" in ids
    assert data["providers"][0]["models"][0]["name"] == "qwen3:8b"
    assert data["default"] == {"provider": "ollama", "model": "qwen3:8b"}


def test_models_explains_when_ollama_is_down(client, monkeypatch):
    def down():
        raise OSError("connection refused")
    monkeypatch.setattr(server, "ollama_models", down)
    data = client.get("/api/models").json()
    ollama = data["providers"][0]
    assert not ollama["available"] and "ollama serve" in ollama["note"]
    assert data["default"]["provider"] == "extractive"      # still usable without any LLM


def test_ask_streams_sources_tokens_then_a_checked_answer(client, monkeypatch):
    monkeypatch.setattr(generate, "stream_llm", lambda *a: iter(["100 ", "[1]", "."]))
    resp = client.post("/api/ask", json={"question": "How many participants?", "provider": "ollama",
                                         "model": "qwen3:8b", "paper_id": "p1"})
    evs = events(resp)
    assert [e["type"] for e in evs] == ["sources", "token", "token", "token", "final"]
    assert len(evs[0]["chunks"]) == 2
    final = evs[-1]["answer"]
    assert not final["refused"] and final["cited"] == [1]
    assert "[Learning Autocomplete Systems as a Communication Game, Experiments > User study]" in final["text"]


def test_ask_runs_the_citation_repair_turn(client, monkeypatch):
    monkeypatch.setattr(generate, "stream_llm", lambda *a: iter(["About 100 people."]))
    monkeypatch.setattr(generate, "call_llm", lambda *a, **kw: "About 100 people [1].")
    evs = events(client.post("/api/ask", json={"question": "How many?", "provider": "ollama", "model": "qwen3:8b"}))
    assert "repair" in [e["type"] for e in evs]
    assert evs[-1]["answer"]["cite_repair"] and not evs[-1]["answer"]["refused"]


def test_off_topic_is_refused_before_any_llm_call(client, monkeypatch):
    def boom(*a):
        raise AssertionError("the LLM should not be called")
    monkeypatch.setattr(generate, "stream_llm", boom)
    evs = events(client.post("/api/ask", json={"question": "What is the capital of France?",
                                               "provider": "ollama", "model": "qwen3:8b"}))
    final = evs[-1]["answer"]
    assert final["refused"] and final["reason"] == "low_relevance"


def test_ask_rejects_models_that_are_not_installed(client):
    resp = client.post("/api/ask", json={"question": "q", "provider": "ollama", "model": "llama-9000"})
    assert resp.status_code == 400 and "isn't installed" in resp.json()["detail"]
    resp = client.post("/api/ask", json={"question": "q", "provider": "ollama", "model": "qwen3:8b",
                                         "mode": "telepathy"})
    assert resp.status_code == 400


def test_llm_errors_reach_the_ui_as_an_error_event(client, monkeypatch):
    def broken(*a):
        raise RuntimeError("model ran out of memory")
        yield  # pragma: no cover  (makes this a generator)
    monkeypatch.setattr(generate, "stream_llm", broken)
    evs = events(client.post("/api/ask", json={"question": "How many?", "provider": "ollama", "model": "qwen3:8b"}))
    assert evs[-1]["type"] == "error" and "out of memory" in evs[-1]["message"]
    assert evs[-1]["request_id"]                              # so the failure can be found in the logs


def test_think_tags_are_stripped_even_when_split_across_chunks():
    pieces = ["<thi", "nk>let me reason", " about it</th", "ink>The answer", " is 100 [1]."]
    assert "".join(generate._strip_think(iter(pieces))) == "The answer is 100 [1]."


def test_the_page_is_served(client):
    html = client.get("/").text
    assert 'id="model"' in html and "/static/app.js" in html
    assert client.get("/static/app.js").status_code == 200
