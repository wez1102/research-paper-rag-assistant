"""
Smoke tests. The first group is pure logic and runs in a second with no models.
The last group needs the real index (python ingest.py) and is skipped otherwise.

    pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from evaluate import first_hit_rank, summarize, token_f1  # noqa: E402
from generate import REFUSAL, extractive_answer, is_refusal, render_citations  # noqa: E402
from ingest import Paragraph, chunk_paper  # noqa: E402
from retrieve import RetrievedChunk, reciprocal_rank_fusion, split_passages, tokenize  # noqa: E402


class WordTokenizer:
    """Stand-in for the HF tokenizer: one token per word, with character offsets."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        offsets, pos = [], 0
        for word in text.split():
            start = text.index(word, pos)
            offsets.append((start, start + len(word)))
            pos = start + len(word)
        return {"offset_mapping": offsets}


def make_chunk(i: int, paper="p1", paras=(0,), section="Intro", text="some text") -> RetrievedChunk:
    return RetrievedChunk(f"{paper}-{i:03d}", paper, f"Paper {paper}", section, text, list(paras), 1.0, "dense")


# --- chunking -------------------------------------------------------------

def test_chunks_respect_size_overlap_and_sections():
    words = " ".join(f"w{i}" for i in range(250))
    paras = [Paragraph("p1", "T", "Intro", 0, words),
             Paragraph("p1", "T", "Method", 1, "short method paragraph")]
    chunks = chunk_paper(paras, WordTokenizer(), size=100, overlap=20)

    intro = [c for c in chunks if c.section == "Intro"]
    assert all(c.n_tokens <= 100 for c in chunks)
    assert len(intro) == 3                                    # 250 words, 100-word windows, 20 overlap
    # The last 20 words of one chunk are the first 20 of the next.
    assert intro[0].text.split()[-20:] == intro[1].text.split()[:20]
    # Sections are never mixed inside a chunk.
    assert [c.section for c in chunks].count("Method") == 1
    assert "method" not in " ".join(c.text for c in intro)


def test_chunks_never_cross_papers():
    paras_a = [Paragraph("a", "A", "Intro", 0, "alpha " * 30)]
    paras_b = [Paragraph("b", "B", "Intro", 0, "beta " * 30)]
    chunks = chunk_paper(paras_a, WordTokenizer(), 50, 10) + chunk_paper(paras_b, WordTokenizer(), 50, 10)
    assert {c.paper_id for c in chunks} == {"a", "b"}
    assert all(("alpha" in c.text) != ("beta" in c.text) for c in chunks)


# --- retrieval helpers ----------------------------------------------------

def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    ids = [cid for cid, _ in fused]
    assert set(ids[:2]) == {"a", "b"}          # both lists like a and b
    assert ids.index("c") > ids.index("b")


def test_weighted_rrf_follows_weights():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60, weights=[1.0, 0.25])
    assert [cid for cid, _ in fused][0] == "a"     # the heavier list wins the tie


def test_split_passages():
    tiny, normal = "Too short.", " ".join(["word"] * 60)
    long_para = " ".join(f"w{i}" for i in range(400))
    parts = split_passages(f"{tiny}\n\n{normal}\n\n{long_para}")
    assert parts[0].startswith("Too short.")        # tiny fragment glued onto its neighbour
    assert all(len(p.split()) <= 160 for p in parts)
    # every word of the long paragraph survives the windowing
    assert {f"w{i}" for i in range(400)} <= set(" ".join(parts).split())


def test_tokenize_keeps_model_names():
    assert "bert-base" in tokenize("We fine-tune BERT-base on SQuAD.")
    assert "the" not in tokenize("the model")


# --- metrics --------------------------------------------------------------

def test_hit_rank_and_summary():
    q = {"paper_id": "p1", "gold_paragraph_ids": [7]}
    results = [make_chunk(0, paras=(1,)), make_chunk(1, paper="p2", paras=(7,)), make_chunk(2, paras=(6, 7))]
    # The p2 chunk has paragraph 7 too, but from the wrong paper, so it must not count.
    assert first_hit_rank(results, q) == 3
    row = summarize([1, 3, None, 2], [0.1, 0.2, 0.3, 0.4])
    assert row["hit@1"] == 0.25 and row["hit@3"] == 0.75 and row["hit@5"] == 0.75
    assert abs(row["mrr"] - (1 + 1 / 3 + 1 / 2) / 4) < 1e-9


def test_token_f1():
    assert token_f1("the BERT model", "BERT model") == 1.0
    assert token_f1("cats", "dogs") == 0.0


# --- citations & refusals -------------------------------------------------

def test_citations_rendered_and_invalid_ones_dropped():
    chunks = [make_chunk(0, section="Intro"), make_chunk(1, section="Results")]
    text, cited, total, valid = render_citations("It works [2]. It is fast [1, 2]. Made up [9].", chunks)
    assert "[Paper p1, Results]" in text and "[Paper p1, Intro]" in text
    assert "[9]" not in text
    assert cited == [2, 1] and total == 4 and valid == 3


def test_refusal_detection_and_extractive_fallback():
    assert is_refusal("I can’t find this in the papers.")
    chunks = [make_chunk(0, text="We evaluate on the SQuAD dataset and report exact match.")]
    assert extractive_answer("Which dataset do they evaluate on?", chunks).endswith("[1]")
    assert extractive_answer("zebra giraffe", chunks) == REFUSAL


def test_uncited_answer_gets_one_repair_turn(monkeypatch):
    import generate
    chunks = [make_chunk(0, section="Results", text="We compare against RoBERTa.")]
    calls = []

    def fake_llm(provider, model, system, user, history=None):
        calls.append(history)
        # First answer forgets the [n]; the repair turn adds it.
        return "RoBERTa BIBREF2." if history is None else "RoBERTa [1]."

    monkeypatch.setattr(generate, "call_llm", fake_llm)
    ans = generate.generate_answer("What is the previous state of the art?", chunks, provider="ollama")
    assert not ans.refused and ans.cited == [1] and ans.timings["cite_repair"]
    assert len(calls) == 2 and calls[1][-1]["content"] == generate.REPAIR_PROMPT

    # If the repair turn still cites nothing, it is refused - no citation is made up.
    monkeypatch.setattr(generate, "call_llm", lambda *a, **k: "RoBERTa BIBREF2.")
    assert generate.generate_answer("q", chunks, provider="ollama").refused


# --- end to end (needs the built index) -----------------------------------

INDEX_READY = (settings.chroma_dir / "chroma.sqlite3").exists()


@pytest.mark.skipif(not INDEX_READY, reason="run `python ingest.py` first")
def test_all_modes_return_k_chunks():
    from retrieve import MODES, Retriever
    retriever = Retriever()
    for mode in MODES:
        results = retriever.retrieve("What datasets do they use?", mode=mode, k=5)
        assert len(results) == 5
        assert all(r.title and r.section and r.text for r in results)
    # Paper filter keeps everything inside one paper.
    pid = retriever.papers()[0][0]
    assert {r.paper_id for r in retriever.retrieve("results", mode="hybrid", paper_id=pid)} == {pid}
