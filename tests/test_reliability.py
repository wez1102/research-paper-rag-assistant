"""
Failure handling: what happens when things around the app go wrong.

The Ollama tests talk to a tiny fake Ollama server over a real socket, so they
exercise the actual HTTP client code (timeouts, broken streams, bad JSON), not a
mock of it. Nothing here needs the index, the models or a real Ollama.

    pytest -q tests/test_reliability.py
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import generate
import server
from generate import LLMBadResponse, LLMTimeout, LLMUnavailable
from retrieve import RetrievedChunk
from tests.test_server import FakeRetriever, events


# ---------------------------------------------------------------------------
# A scripted fake Ollama
# ---------------------------------------------------------------------------

def _line(content: str = "", done: bool = False) -> bytes:
    return (json.dumps({"message": {"role": "assistant", "content": content}, "done": done}) + "\n").encode()


SCRIPTS = {
    # name -> list of (bytes to send, seconds to sleep after sending)
    "ok": [(_line("Yes "), 0), (_line("[1]."), 0), (_line(done=True), 0)],
    "malformed": [(_line("Yes "), 0), (b"{not json\n", 0)],
    "error_line": [(b'{"error": "model requires more system memory"}\n', 0)],
    "cut_off": [(_line("Yes, they "), 0)],                     # connection closes, no done=true
    "stall": [(_line("Yes "), 3.0), (_line(done=True), 0)],   # goes quiet longer than the read timeout
}


class FakeOllama(BaseHTTPRequestHandler):
    script = "ok"

    def log_message(self, *args):          # keep pytest output clean
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/api/chat" and FakeOllama.script == "http500":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")
            return
        if self.path == "/api/chat" and FakeOllama.script == "nonstream_garbage":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"unexpected": true}')
            return
        if self.path == "/api/chat" and FakeOllama.script == "nonstream_ok":
            body = json.dumps({"message": {"content": "<think>hm</think>Yes [1]."}, "done": True}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for data, pause in SCRIPTS[FakeOllama.script]:
            try:
                self.wfile.write(data)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            time.sleep(pause)


@pytest.fixture()
def fake_ollama(override_settings):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    override_settings(ollama_host=f"http://127.0.0.1:{srv.server_address[1]}", llm_read_timeout=1.0,
                      llm_connect_timeout=1.0, llm_total_timeout=10.0)

    def use(script: str):
        FakeOllama.script = script

    yield use
    srv.shutdown()


def _stream() -> str:
    return "".join(generate.stream_llm("ollama", "qwen3:8b", "system", "user"))


def test_complete_stream_is_returned(fake_ollama):
    fake_ollama("ok")
    assert _stream() == "Yes [1]."


def test_malformed_line_is_reported_not_swallowed(fake_ollama):
    fake_ollama("malformed")
    with pytest.raises(LLMBadResponse):
        _stream()


def test_error_line_from_ollama_is_reported(fake_ollama):
    fake_ollama("error_line")
    with pytest.raises(LLMBadResponse, match="memory"):
        _stream()


def test_stream_cut_off_before_done_is_an_error(fake_ollama):
    # Without the done=true marker the answer is incomplete and must not look finished.
    fake_ollama("cut_off")
    with pytest.raises(LLMTimeout):
        _stream()


def test_model_that_goes_quiet_times_out(fake_ollama):
    fake_ollama("stall")
    t0 = time.perf_counter()
    with pytest.raises(LLMTimeout):
        _stream()
    assert time.perf_counter() - t0 < 2.5               # read timeout is 1 s, not the 3 s stall


def test_total_time_cap(fake_ollama, override_settings):
    # Each piece arrives in time, but the whole answer exceeds the overall cap.
    fake_ollama("stall")
    override_settings(llm_read_timeout=5.0, llm_total_timeout=0.5)
    with pytest.raises(LLMTimeout):
        _stream()


def test_http_error_status(fake_ollama):
    fake_ollama("http500")
    with pytest.raises(LLMBadResponse, match="500"):
        _stream()
    with pytest.raises(LLMBadResponse):
        generate.call_llm("ollama", "qwen3:8b", "s", "u")


def test_non_streaming_call_checks_the_reply_shape(fake_ollama):
    fake_ollama("nonstream_garbage")
    with pytest.raises(LLMBadResponse):
        generate.call_llm("ollama", "qwen3:8b", "s", "u")
    fake_ollama("nonstream_ok")
    assert generate.call_llm("ollama", "qwen3:8b", "s", "u") == "Yes [1]."     # think tags stripped


def test_ollama_not_running(override_settings):
    # Grab a free port and close it again, so nothing is listening there.
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    override_settings(ollama_host=f"http://127.0.0.1:{port}", llm_connect_timeout=0.5)
    with pytest.raises(LLMUnavailable):
        _stream()
    with pytest.raises(LLMUnavailable):
        generate.call_llm("ollama", "qwen3:8b", "s", "u")       # retried once, then gives up
    with pytest.raises(LLMUnavailable):
        generate.ollama_models()


# ---------------------------------------------------------------------------
# Closing the stream early (the browser went away) stops generation
# ---------------------------------------------------------------------------

def test_closing_the_answer_stream_closes_the_llm_stream(monkeypatch):
    closed = []

    def slow_llm(*a):
        try:
            for word in ["one ", "two ", "three ", "[1]."]:
                yield word
        finally:
            closed.append(True)        # runs when the consumer goes away

    monkeypatch.setattr(generate, "stream_llm", slow_llm)
    stream = generate.answer_events("How many?", FakeRetriever(), provider="ollama", model="qwen3:8b")
    assert next(stream)["type"] == "sources"
    assert next(stream)["type"] == "token"
    stream.close()
    assert closed == [True]


# ---------------------------------------------------------------------------
# Server: readiness, missing index, auth, tracing, back-pressure
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient

    server.state["retriever"] = FakeRetriever()
    server.state["load_error"] = None
    server._gen_slots = None
    monkeypatch.setattr(server, "ollama_models", lambda: [{"name": "qwen3:8b"}])
    monkeypatch.setattr(server, "resolve_provider", lambda: "ollama")
    monkeypatch.setattr(server, "model_for", lambda p: "qwen3:8b")
    return TestClient(server.app)


ASK = {"question": "How many participants?", "provider": "ollama", "model": "qwen3:8b"}


def test_health_and_readiness(client, monkeypatch):
    assert client.get("/healthz").json()["status"] == "ok"
    ready = client.get("/readyz")
    assert ready.status_code == 200 and ready.json()["status"] == "ready"

    # No LLM: still ready (quotes still work), but flagged as degraded.
    def down():
        raise LLMUnavailable("connection refused")
    monkeypatch.setattr(server, "ollama_models", down)
    ready = client.get("/readyz")
    assert ready.status_code == 200 and ready.json()["status"] == "degraded"
    assert ready.json()["checks"]["ollama"]["ok"] is False


def test_missing_index_gives_503_not_a_crash(client):
    server.state["retriever"] = None
    server.state["load_error"] = "IndexNotReady: Collection 'qasper_chunks_450' not found"
    assert client.get("/healthz").status_code == 200           # process is alive...
    ready = client.get("/readyz")
    assert ready.status_code == 503 and "not found" in ready.json()["checks"]["index"]["error"]
    resp = client.post("/api/ask", json=ASK)                    # ...but can't answer
    assert resp.status_code == 503 and "not found" in resp.json()["detail"]
    assert client.get("/api/info").status_code == 503


def test_load_failure_is_recorded(monkeypatch):
    from retrieve import IndexNotReady

    def broken(**kw):
        raise IndexNotReady("Collection 'x' not found. Run `python ingest.py` first.")
    monkeypatch.setattr(server, "Retriever", broken)
    server.load_retriever()
    assert server.state["retriever"] is None and "ingest.py" in server.state["load_error"]


def test_api_key_auth(client, override_settings, monkeypatch):
    monkeypatch.setattr(generate, "stream_llm", lambda *a: iter(["100 [1]."]))
    override_settings(api_keys=("s3cret",))
    assert client.get("/api/info").status_code == 401
    assert client.get("/api/info", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/api/info", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.post("/api/ask", json=ASK).status_code == 401
    ok = client.post("/api/ask", json=ASK, headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200 and events(ok)[-1]["type"] == "final"
    # Health checks stay open so load balancers / uptime checks work without a key.
    assert client.get("/healthz").status_code == 200 and client.get("/readyz").status_code == 200


def test_request_id_is_echoed_and_carried_into_events(client, monkeypatch):
    monkeypatch.setattr(generate, "stream_llm", lambda *a: iter(["100 [1]."]))
    resp = client.post("/api/ask", json=ASK, headers={"X-Request-ID": "trace-123"})
    assert resp.headers["x-request-id"] == "trace-123"
    evs = events(resp)
    assert evs[0]["request_id"] == "trace-123" and evs[-1]["answer"]["request_id"] == "trace-123"
    assert "rerank_s" in evs[0]["timings"] or evs[0]["timings"] == {}
    # One is made up when the caller doesn't send one.
    assert len(client.get("/api/info").headers["x-request-id"]) == 16


def test_busy_server_says_so(client, override_settings):
    override_settings(max_concurrent_generations=0, queue_timeout=0.2)
    server._gen_slots = None
    resp = client.post("/api/ask", json=ASK)
    assert resp.status_code == 503 and resp.headers["retry-after"] == "10"


def test_concurrent_requests_all_complete(monkeypatch, override_settings):
    """Several questions at once: every one gets a complete, correctly ordered stream."""
    from fastapi.testclient import TestClient

    def slow_llm(*a):
        for piece in ["About ", "100 ", "[1]."]:
            time.sleep(0.05)
            yield piece

    monkeypatch.setattr(generate, "stream_llm", slow_llm)
    monkeypatch.setattr(server, "load_retriever", lambda: server.state.update(retriever=FakeRetriever(),
                                                                              load_error=None))
    monkeypatch.setattr(server, "ollama_models", lambda: [{"name": "qwen3:8b"}])
    override_settings(max_concurrent_generations=2, queue_timeout=10)
    server._gen_slots = None
    results = []
    with TestClient(server.app) as c:                 # one event loop shared by all threads
        def ask(i):
            r = c.post("/api/ask", json=ASK, headers={"X-Request-ID": f"req-{i}"})
            results.append((i, r.status_code, events(r)))
        threads = [threading.Thread(target=ask, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
    assert len(results) == 6
    for i, status, evs in results:
        assert status == 200
        assert [e["type"] for e in evs] == ["sources", "token", "token", "token", "final"]
        assert evs[-1]["answer"]["request_id"] == f"req-{i}" and not evs[-1]["answer"]["refused"]


def test_llm_errors_have_a_code(client, monkeypatch):
    def down(*a):
        raise LLMUnavailable("Couldn't reach Ollama")
        yield  # pragma: no cover
    monkeypatch.setattr(generate, "stream_llm", down)
    evs = events(client.post("/api/ask", json=ASK))
    assert evs[-1]["type"] == "error" and evs[-1]["code"] == "llm_unavailable"


def test_retriever_rejects_an_index_from_another_embedding_model(tmp_path, override_settings):
    """A query embedded with model A against vectors from model B returns junk silently - refuse instead."""
    import chromadb

    from retrieve import IndexNotReady, Retriever

    override_settings(chroma_dir=tmp_path)
    client = chromadb.PersistentClient(path=str(tmp_path))
    col = client.create_collection("qasper_chunks_450", metadata={"hnsw:space": "cosine",
                                                                  "embed_model": "some-other-model",
                                                                  "n_chunks": 1})
    col.add(ids=["p-000"], documents=["text"], embeddings=[[0.1, 0.2]],
            metadatas=[{"paper_id": "p", "title": "T", "section": "S", "chunk_id": "p-000",
                        "paragraph_ids": "0", "n_tokens": 1}])
    with pytest.raises(IndexNotReady, match="some-other-model"):
        Retriever(load_reranker=False)
    client.create_collection("empty")
    with pytest.raises(IndexNotReady, match="empty"):
        Retriever(collection_name="empty", load_reranker=False)
    with pytest.raises(IndexNotReady, match="not found"):
        Retriever(collection_name="missing", load_reranker=False)


def test_render_ignores_reference_markers_from_the_paper():
    """Papers' own BIBREF markers aren't our citations and mustn't count as valid ones."""
    chunks = [RetrievedChunk("p-000", "p", "T", "S", "x", [0], 1.0, "rerank")]
    text, cited, total, valid = generate.render_citations("RoBERTa BIBREF2 is used [1].", chunks)
    assert cited == [1] and total == 1 and valid == 1
