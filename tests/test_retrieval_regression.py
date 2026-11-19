"""
Retrieval regression guard. Re-scores the app split (144 questions) with the real
models and index and fails if quality drops below the numbers recorded in
tests/golden/retrieval_app.json (measured in round 2, exact dense search).

A small tolerance (default 1 question per metric) absorbs floating-point differences
between CPUs (x86 vs ARM can flip near-ties in the cross-encoder). Anything bigger is a
real change - if it's an intended improvement, re-record the golden file:

    python tests/test_retrieval_regression.py --record

Needs the built index (python ingest.py) and the models; skipped otherwise.
The reranked mode runs on the first RERANK_SUBSET questions to keep CI under ~5 minutes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402

GOLDEN = ROOT / "tests" / "golden" / "retrieval_app.json"
RERANK_SUBSET = 40
INDEX_READY = (settings.chroma_dir / "chroma.sqlite3").exists() and settings.questions_path("app").exists()


def measure() -> dict:
    from evaluate import first_hit_rank, load_questions
    from retrieve import Retriever

    questions = load_questions("app")
    r = Retriever()
    out = {}
    for mode in ("dense", "hybrid", "hybrid_rerank"):
        qs = questions[:RERANK_SUBSET] if mode == "hybrid_rerank" else questions
        for scope in ("paper", "corpus"):
            hits = 0
            for q in qs:
                res = r.retrieve(q["question"], mode=mode, paper_id=q["paper_id"] if scope == "paper" else None)
                rank = first_hit_rank(res, q)
                hits += rank is not None and rank <= 5
            out[f"{mode}/{scope}"] = {"hits@5": hits, "n": len(qs)}
    return out


@pytest.mark.skipif(not INDEX_READY, reason="run `python ingest.py` first")
@pytest.mark.skipif(not GOLDEN.exists(), reason="no golden file recorded yet")
def test_retrieval_quality_has_not_dropped():
    golden = json.loads(GOLDEN.read_text())
    tolerance = golden.get("tolerance_questions", 1)
    now = measure()
    drops = []
    for key, ref in golden["metrics"].items():
        got = now[key]
        assert got["n"] == ref["n"], f"{key}: question set changed ({got['n']} vs {ref['n']})"
        if got["hits@5"] < ref["hits@5"] - tolerance:
            drops.append(f"{key}: {got['hits@5']}/{got['n']} hits@5, recorded {ref['hits@5']}")
    assert not drops, "Retrieval got worse:\n" + "\n".join(drops)


if __name__ == "__main__" and "--record" in sys.argv:
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    from evaluate import config_fingerprint
    GOLDEN.write_text(json.dumps({"config": config_fingerprint(), "tolerance_questions": 1,
                                  "rerank_subset": RERANK_SUBSET, "metrics": measure()}, indent=2))
    print(f"recorded {GOLDEN}")
