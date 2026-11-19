"""
Indexing must be repeatable: running ingest twice gives the same index, never a
second copy of the records, and a crash halfway leaves the previous index usable.
Uses a fake embedder and a temporary Chroma folder, so no models are needed.

    pytest -q tests/test_ingest.py
"""

from __future__ import annotations

import numpy as np
import pytest

import ingest
from ingest import Chunk, Paragraph, build_index, chunk_paper, index_manifest
from tests.test_smoke import WordTokenizer


class FakeEmbedder:
    """Deterministic 8-d 'embeddings' from a hash of the text."""

    def __init__(self, fail_after: int | None = None):
        self.fail_after = fail_after

    def encode(self, texts, **kw):
        if self.fail_after is not None and len(texts) > self.fail_after:
            raise RuntimeError("simulated crash while embedding")
        rng = [np.random.default_rng(abs(hash(t)) % (2 ** 32)).random(8) for t in texts]
        v = np.array(rng, dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def make_chunks(texts: list[str]) -> list[Chunk]:
    return [Chunk(f"p1-{i:03d}", "p1", "Paper", "Intro", t, [i], len(t.split())) for i, t in enumerate(texts)]


@pytest.fixture()
def chroma(tmp_path, override_settings):
    import chromadb
    override_settings(chroma_dir=tmp_path)
    return lambda: chromadb.PersistentClient(path=str(tmp_path))


def test_second_build_is_a_no_op(chroma):
    chunks = make_chunks(["alpha beta", "gamma delta", "epsilon"])
    m = index_manifest(chunks, 450, 50)
    assert build_index(chunks, "col", FakeEmbedder(), m) is True
    assert build_index(chunks, "col", FakeEmbedder(), m) is False          # skipped, nothing changed
    col = chroma().get_collection("col")
    assert col.count() == 3 and col.metadata["manifest_hash"] == m["manifest_hash"]


def test_changed_content_rebuilds_without_duplicates(chroma):
    first = make_chunks(["alpha beta", "gamma delta", "epsilon"])
    build_index(first, "col", FakeEmbedder(), index_manifest(first, 450, 50))
    second = make_chunks(["alpha beta", "gamma delta CHANGED"])
    m2 = index_manifest(second, 450, 50)
    assert m2["manifest_hash"] != index_manifest(first, 450, 50)["manifest_hash"]
    assert build_index(second, "col", FakeEmbedder(), m2) is True
    col = chroma().get_collection("col")
    got = col.get()
    assert col.count() == 2 and sorted(got["ids"]) == ["p1-000", "p1-001"]
    assert "CHANGED" in got["documents"][got["ids"].index("p1-001")]


def test_force_rebuild_keeps_exactly_one_copy(chroma):
    chunks = make_chunks(["a b c", "d e f"])
    m = index_manifest(chunks, 450, 50)
    for _ in range(3):
        build_index(chunks, "col", FakeEmbedder(), m, force=True)
    assert chroma().get_collection("col").count() == 2
    assert [c.name for c in chroma().list_collections()] == ["col"]          # no leftover temp collection


def test_crash_mid_build_leaves_the_old_index_intact(chroma):
    old = make_chunks(["one", "two"])
    build_index(old, "col", FakeEmbedder(), index_manifest(old, 450, 50))
    new = make_chunks(["one", "two", "three", "four"])
    with pytest.raises(RuntimeError, match="simulated crash"):
        build_index(new, "col", FakeEmbedder(fail_after=2), index_manifest(new, 450, 50))
    col = chroma().get_collection("col")
    assert col.count() == 2 and col.metadata["manifest_hash"] == index_manifest(old, 450, 50)["manifest_hash"]
    # ...and the next successful run cleans up the half-built temp collection.
    build_index(new, "col", FakeEmbedder(), index_manifest(new, 450, 50))
    assert [c.name for c in chroma().list_collections()] == ["col"]
    assert chroma().get_collection("col").count() == 4


def test_duplicate_chunk_ids_are_refused(chroma):
    dup = make_chunks(["a", "b"])
    dup[1].chunk_id = dup[0].chunk_id
    with pytest.raises(ValueError, match="Duplicate"):
        build_index(dup, "col", FakeEmbedder(), index_manifest(dup, 450, 50))


def test_chunking_is_deterministic_and_ids_are_unique():
    paras = [Paragraph("p1", "T", "Intro", 0, " ".join(f"w{i}" for i in range(300))),
             Paragraph("p1", "T", "Method", 1, "short method paragraph")]
    a = chunk_paper(paras, WordTokenizer(), 100, 20)
    b = chunk_paper(paras, WordTokenizer(), 100, 20)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b] and [c.text for c in a] == [c.text for c in b]
    assert len({c.chunk_id for c in a}) == len(a)
    assert index_manifest(a, 100, 20) == index_manifest(b, 100, 20)


def test_manifest_changes_with_the_embedding_model(override_settings):
    chunks = make_chunks(["x"])
    m1 = index_manifest(chunks, 450, 50)
    override_settings(embed_model="BAAI/bge-base-en-v1.5")
    assert index_manifest(chunks, 450, 50)["manifest_hash"] != m1["manifest_hash"]
    # A local copy of the same model is the same model.
    override_settings(embed_model="/models/bge-small-en-v1.5")
    assert index_manifest(chunks, 450, 50)["manifest_hash"] == m1["manifest_hash"]


def test_splits_never_share_papers():
    from config import settings
    ranges = [set(range(s, s + n)) for s, n in settings.splits.values()]
    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            assert not ranges[i] & ranges[j]


def test_unanswerable_questions_are_collected():
    paper = {"id": "p1", "title": "T", "abstract": "We study things.",
             "full_text": {"section_name": ["Intro"], "paragraphs": [["The dataset has 100 rows."]]},
             "qas": {"question_id": ["q1", "q2"], "question": ["How many rows?", "What GPU?"],
                     "answers": [{"answer": [{"unanswerable": False, "extractive_spans": ["100"], "yes_no": None,
                                              "free_form_answer": "", "evidence": ["The dataset has 100 rows."]}]},
                                 {"answer": [{"unanswerable": True, "extractive_spans": [], "yes_no": None,
                                              "free_form_answer": "", "evidence": []}]}]}}
    unans: list = []
    qs, stats = ingest.build_questions(paper, ingest.paper_paragraphs(paper), unans)
    assert [q["question_id"] for q in qs] == ["q1"] and qs[0]["gold_paragraph_ids"] == [1]
    assert [u["question_id"] for u in unans] == ["q2"] and stats["unanswerable"] == 1
