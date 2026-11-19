"""
Tests for the answer-evaluation helpers and for the "which paper?" / refusal rules.
No models, index or LLM needed.

    pytest -q tests/test_answer_eval.py
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

import generate
from answer_eval import messages_sha1, split_claims, summarize_answers
from retrieve import RetrievedChunk
from tests.test_server import FakeRetriever


# --- claims & citations ---------------------------------------------------

def test_split_claims_keeps_each_sentences_citations():
    claims = split_claims("BLEU and ROUGE [2]. They also run a human study [1][3]. It is fast.")
    assert [c["cites"] for c in claims] == [[2], [1, 3], []]
    assert claims[0]["text"] == "BLEU and ROUGE"


def test_citation_after_the_full_stop_belongs_to_that_sentence():
    claims = split_claims("They use BLEU. [2] The corpus is ASPEC [1, 4].")
    assert [c["cites"] for c in claims] == [[2], [1, 4]]


def test_short_answers_are_one_claim():
    assert split_claims("Yes [1].") == [{"text": "Yes", "cites": [1]}]
    assert split_claims("100 [3]") == [{"text": "100", "cites": [3]}]


def test_prompt_hash_matches_the_browser():
    """The relay rebuilds prompts in JS and compares SHA-1s, so both sides must serialise identically."""
    messages = [{"role": "system", "content": "Rules: \"quote\" \\ back\\slash\ttab"},
                {"role": "user", "content": "Arabic أبوك, Japanese 日本語, emoji 🙂, line\nbreak, </script>"}]
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    script = ("const c=require('crypto');const m=JSON.parse(require('fs').readFileSync(0,'utf8'));"
              "process.stdout.write(c.createHash('sha1').update(JSON.stringify(m),'utf8').digest('hex'))")
    out = subprocess.run([node, "-e", script], input=json.dumps(messages), capture_output=True, text=True, check=True)
    assert out.stdout == messages_sha1(messages)


def test_summary_counts():
    import pandas as pd
    df = pd.DataFrame([
        {"category": "answerable", "expected": "answer", "refused": False, "n_claims": 2, "n_claims_cited": 2,
         "support_judge": ["SUPPORTED", "UNSUPPORTED"], "support_review": ["SUPPORTED", "PARTIAL"],
         "correct_judge": "CORRECT", "correct_review": "PARTIAL", "behaviour_review": None,
         "generation_s": 4.0, "ttft_s": 1.0},
        {"category": "unanswerable", "expected": "refuse", "refused": True, "n_claims": 0, "n_claims_cited": 0,
         "support_judge": [], "support_review": [], "correct_judge": None, "correct_review": None,
         "behaviour_review": None, "generation_s": 2.0, "ttft_s": None},
        {"category": "answerable", "expected": "answer", "refused": False, "n_claims": 1, "n_claims_cited": 0,
         "support_judge": [None], "support_review": [None], "correct_judge": "INCORRECT",
         "correct_review": "INCORRECT", "behaviour_review": None, "generation_s": 6.0, "ttft_s": 2.0},
    ])
    s = summarize_answers(df)
    assert s["correct_review"] == 0.0 and s["correct_or_partial_review"] == 0.5
    assert s["claims_supported_judge"] == 0.5 and s["claims_supported_or_partial_review"] == 1.0
    assert s["citation_coverage"] == round(2 / 3, 3) and s["unanswerable_refused"] == "1/1"


# --- "which paper?" and refusals --------------------------------------------

class SpecificityRetriever(FakeRetriever):
    """FakeRetriever that also answers question_specificity like the real one."""

    def question_specificity(self, question):
        specific = "autocomplete" in question.lower()
        return {"specific": specific, "rarest_term": None, "papers_with_term": 1 if specific else 40, "limit": 3}


def _events(question, paper_id=None, monkeypatch=None):
    return list(generate.answer_events(question, SpecificityRetriever(), provider="ollama", model="qwen3:8b",
                                       paper_id=paper_id))


def test_generic_corpus_question_gets_which_paper(monkeypatch):
    monkeypatch.setattr(generate, "stream_llm", lambda *a: pytest.fail("no LLM call expected"))
    final = _events("What was the baseline?")[-1]["answer"]
    assert final["refused"] and final["reason"] == "needs_paper"
    assert final["candidates"][0]["paper_id"] == "p1"


def test_specific_or_paper_scoped_or_cross_paper_questions_are_answered(monkeypatch):
    monkeypatch.setattr(generate, "stream_llm", lambda *a: iter(["100 [1]."]))
    for q, pid in [("How was the autocomplete game evaluated?", None),      # names something specific
                   ("What was the baseline?", "p1"),                       # a paper is picked
                   ("Which papers report a user study?", None)]:           # explicitly about several papers
        final = _events(q, pid)[-1]["answer"]
        assert not final["refused"], q


def test_clarify_can_be_turned_off(monkeypatch, override_settings):
    override_settings(clarify_generic=False)
    monkeypatch.setattr(generate, "stream_llm", lambda *a: iter(["100 [1]."]))
    assert not _events("What was the baseline?")[-1]["answer"]["refused"]


def test_relevance_floor_wins_over_clarify(monkeypatch):
    monkeypatch.setattr(generate, "stream_llm", lambda *a: pytest.fail("no LLM call expected"))
    final = _events("What is the capital of France?")[-1]["answer"]     # FakeRetriever scores this -11
    assert final["reason"] == "low_relevance"


def test_model_refusal_and_uncited_answers_are_refusals():
    chunks = [RetrievedChunk("p-000", "p", "T", "S", "text", [0], 2.0, "rerank")]
    assert generate.finish_answer("q", "I can't find this in the papers.", chunks, "x", "y").reason == "model_declined"
    assert generate.finish_answer("q", "It is 100.", chunks, "x", "y").reason == "no_citations"
    assert generate.finish_answer("q", "It is 100 [7].", chunks, "x", "y").reason == "no_citations"   # [7] invalid
    ok = generate.finish_answer("q", "It is 100 [1].", chunks, "x", "y")
    assert not ok.refused and ok.cited == [1]
