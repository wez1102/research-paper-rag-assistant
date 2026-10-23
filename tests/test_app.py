"""
Runs the Streamlit app headlessly with Streamlit's own testing harness and
checks that asking a question shows an answer and the retrieved-chunks panel.
Needs the built index, so it's skipped until `python ingest.py` has run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402

pytestmark = pytest.mark.skipif(not (settings.chroma_dir / "chroma.sqlite3").exists(),
                                reason="run `python ingest.py` first")


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    # Force the extractive backend: no API key, no cost, deterministic output.
    import generate
    monkeypatch.setattr(generate, "resolve_provider", lambda: "extractive")


def test_app_answers_with_citations():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception

    at.text_input(key="question").input("Which datasets are used for Armenian named entity recognition?")
    next(b for b in at.button if b.label == "Ask").click().run()
    assert not at.exception

    page = " ".join(m.value for m in at.markdown)
    assert "[pioNER: Datasets and Baselines for Armenian Named Entity Recognition," in page
    assert any("Retrieved chunks" in e.label for e in at.expander)


def test_app_refuses_off_topic():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    at.text_input(key="question").input("What is the capital of France?")
    next(b for b in at.button if b.label == "Ask").click().run()
    assert any("can't find this in the papers" in w.value for w in at.warning)
