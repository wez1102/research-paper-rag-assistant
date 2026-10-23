"""
Step 3: write an answer using ONLY the retrieved chunks, with citations.

How citations work:
  The LLM sees the chunks as numbered sources [1]..[5] and is told to put the
  source number after every claim. Small local models are far more reliable at
  writing "[2]" than at copying a 15-word paper title correctly. Afterwards we
  swap each number for "[paper title, section]", so the user gets the required
  format and we can also check that every citation points at a real source.

When do we say "I can't find this in the papers"?
  - the reranker's best score is very low (nothing relevant came back), or
  - the model itself says the sources don't answer the question, or
  - the model answered but cited nothing valid, even after one follow-up turn
    asking it to add source numbers (an uncited answer is a guess).

LLM backends (set LLM_PROVIDER, default "auto"):
  ollama     - free, local (`ollama pull qwen3:8b`)
  openai     - any OpenAI-compatible API (OpenAI, Groq, Together...) via OPENAI_API_KEY
  anthropic  - Claude via ANTHROPIC_API_KEY
  extractive - no LLM at all: quotes the most relevant sentences. Keeps the demo
               working with zero keys and zero cost.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Iterator

from config import settings
from retrieve import RetrievedChunk, Retriever, tokenize, MODES

REFUSAL = "I can't find this in the papers."

# v1 is the original prompt, kept so the two can be compared (PROMPT_VERSION=v1).
PROMPTS = {
    "v1": f"""You answer questions about NLP research papers using ONLY the numbered sources provided.

Rules:
- Every sentence that states a fact must end with the number of the source(s) it came from, like [1] or [2][4].
- Use only information that appears in the sources. Do not use outside knowledge.
- If the sources do not contain the answer, reply with exactly: {REFUSAL}
- Be concise: at most 4 sentences. Answer the question directly first.""",

    # v2 fixes the two things we saw going wrong with v1:
    #  * answers were long paragraphs, when a reader (and QASPER's short reference
    #    answers) wants the fact first - "BLEU", "100", "Yes";
    #  * it refused when a source said the same thing in different words.
    "v2": f"""You answer questions about NLP research papers using ONLY the numbered sources provided.

How to answer:
1. Start with the shortest direct answer: a name, a number, a short list, or "Yes"/"No" for a yes/no question. Use the exact terms the sources use.
2. Optionally add one or two short sentences of supporting detail.
3. End every sentence that states a fact with its source number(s), like [1] or [2][4].

Rules:
- Use only the sources. Do not use outside knowledge.
- Sources often use different words than the question (synonyms, abbreviations, paraphrases). If a source answers the question in other words, that counts - answer it.
- Only if none of the sources is relevant to the question, reply with exactly: {REFUSAL}""",
}
SYSTEM_PROMPT = PROMPTS[settings.prompt_version]

# Sent as a follow-up turn when the model answered but cited nothing we can use.
# On the dev papers this happened when the model copied the paper's own reference
# marker ("based on RoBERTa BIBREF2.") instead of a source number. Before, that whole
# (correct) answer was thrown away as a refusal. Now the model gets one chance to
# point at a source - or to admit it can't. We never add citations ourselves.
REPAIR_PROMPT = ("Your answer has no source numbers. Rewrite it so that every fact ends with the number "
                 "of the source it came from, like [1] or [2][4]. Use only the numbered sources above. "
                 f"If none of them supports the answer, reply with exactly: {REFUSAL}")

_CITE_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


@dataclass
class Answer:
    question: str
    text: str                    # final answer with [title, section] citations
    raw: str                     # what the model actually wrote (with [n] markers)
    refused: bool
    cited: list[int]             # 1-based source numbers that were cited (valid ones)
    citations_total: int         # how many citation markers the model wrote
    citations_valid: int         # ...and how many pointed at a real retrieved chunk
    provider: str
    model: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    # Why it was refused, so the UI can say something more useful than "no":
    # "low_relevance" (reranker floor), "model_declined", "no_citations", or "".
    reason: str = ""

    @property
    def citation_accuracy(self) -> float | None:
        return self.citations_valid / self.citations_total if self.citations_total else None


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] Paper: {c.title}\nSection: {c.section}\n{c.text}")
    sources = "\n\n---\n\n".join(blocks)
    return f"Sources:\n\n{sources}\n\n---\n\nQuestion: {question}\nAnswer (with [n] citations):"


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

def _ollama_up() -> bool:
    try:
        with urllib.request.urlopen(f"{settings.ollama_host}/api/tags", timeout=1):
            return True
    except Exception:
        return False


def ollama_models() -> list[dict]:
    """
    The chat models installed in the local Ollama, for the model picker in the UI.
    Raises if Ollama isn't reachable, so the caller can tell "not running" apart
    from "running but nothing pulled yet".
    """
    with urllib.request.urlopen(f"{settings.ollama_host}/api/tags", timeout=2) as resp:
        tags = json.loads(resp.read()).get("models", [])
    models = []
    for m in tags:
        details = m.get("details") or {}
        family = (details.get("family") or "").lower()
        # Embedding models (nomic-embed-text, bge-m3...) show up here too but can't chat.
        if "embed" in m["name"].lower() or family in {"bert", "nomic-bert"}:
            continue
        models.append({
            "name": m["name"],
            "size_gb": round(m.get("size", 0) / 1e9, 1),
            "params": details.get("parameter_size", ""),
            "quant": details.get("quantization_level", ""),
        })
    return sorted(models, key=lambda m: m["name"])


def resolve_provider() -> str:
    """Pick a backend. 'auto' goes for whatever is available, cheapest-to-set-up last."""
    p = settings.llm_provider.lower()
    if p != "auto":
        return p
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if _ollama_up():
        return "ollama"
    return "extractive"


def model_for(provider: str) -> str:
    return settings.llm_model or settings.default_models.get(provider, "")


def call_llm(provider: str, model: str, system: str, user: str, history: list[dict] | None = None) -> str:
    """
    Send one prompt to the chosen provider. Temperature 0 so answers are repeatable.
    `history` holds any extra turns after the first user message (used by the
    citation repair step: the model's first answer, then the repair request).
    """
    turns = [{"role": "user", "content": user}] + (history or [])
    if provider == "ollama":
        # Plain HTTP keeps us free of yet another client library.
        payload = {
            "model": model,
            "stream": False,
            # Reasoning models (Qwen3, DeepSeek-R1...) would otherwise "think out loud"
            # first, which is slow and pollutes the answer. Plain models ignore this.
            "think": False,
            "messages": [{"role": "system", "content": system}] + turns,
            "options": {"temperature": 0, "seed": settings.seed, "num_predict": settings.max_answer_tokens},
        }
        req = urllib.request.Request(f"{settings.ollama_host}/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            content = json.loads(resp.read())["message"]["content"]
        # Older Ollama versions ignore "think" and inline the reasoning instead.
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(base_url=settings.openai_base_url or None)
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=settings.max_answer_tokens,
            messages=[{"role": "system", "content": system}] + turns,
        )
        return resp.choices[0].message.content or ""

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        # Note: the 2.x SDK dropped the `temperature` argument, so we just leave it out.
        resp = client.messages.create(
            model=model,
            max_tokens=settings.max_answer_tokens,
            system=system,
            messages=turns,
        )
        return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")

    raise ValueError(f"Unknown LLM provider '{provider}'")


def _strip_think(pieces: Iterator[str]) -> Iterator[str]:
    """
    Drop <think>...</think> from a token stream. Ollama honours think=False for
    Qwen3, but older versions inline the reasoning, and the tags can be split
    across chunks - so hold back anything that could be the start of a tag.
    """
    buf, inside = "", False
    for piece in pieces:
        buf += piece
        while True:
            if inside:
                end = buf.find("</think>")
                if end < 0:
                    buf = buf[-8:]          # keep a tail in case "</think>" is split
                    break
                buf, inside = buf[end + 8:], False
            else:
                start = buf.find("<think>")
                if start >= 0:
                    if start:
                        yield buf[:start]
                    buf, inside = buf[start + 7:], True
                    continue
                safe = len(buf) - 6         # "<think" could still be forming at the end
                if safe > 0:
                    yield buf[:safe]
                    buf = buf[safe:]
                break
    if buf and not inside:
        yield buf


def stream_llm(provider: str, model: str, system: str, user: str) -> Iterator[str]:
    """
    Same as call_llm, but yields the answer piece by piece as the model writes it,
    so the web UI can show text immediately instead of after 10 seconds of nothing.
    Joining the pieces gives the same text call_llm would have returned.
    """
    if provider == "ollama":
        payload = {
            "model": model, "stream": True, "think": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0, "seed": settings.seed, "num_predict": settings.max_answer_tokens},
        }
        req = urllib.request.Request(f"{settings.ollama_host}/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})

        def pieces() -> Iterator[str]:
            # Ollama streams one JSON object per line.
            with urllib.request.urlopen(req, timeout=120) as resp:
                for line in resp:
                    if line.strip():
                        msg = json.loads(line)
                        if msg.get("error"):
                            raise RuntimeError(msg["error"])
                        yield msg.get("message", {}).get("content", "")

        yield from _strip_think(pieces())
        return

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(base_url=settings.openai_base_url or None)
        stream = client.chat.completions.create(
            model=model, temperature=0, max_tokens=settings.max_answer_tokens, stream=True,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        for event in stream:
            if event.choices and event.choices[0].delta.content:
                yield event.choices[0].delta.content
        return

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        with client.messages.stream(model=model, max_tokens=settings.max_answer_tokens, system=system,
                                    messages=[{"role": "user", "content": user}]) as stream:
            yield from stream.text_stream
        return

    raise ValueError(f"Unknown LLM provider '{provider}'")


def extractive_answer(question: str, chunks: list[RetrievedChunk], max_sentences: int = 2) -> str:
    """
    The no-LLM fallback. Splits the retrieved chunks into sentences, scores each by
    how many question words it shares (weighted towards rarer words), and returns
    the top couple with their source numbers. Crude, but always grounded.
    """
    q_terms = set(tokenize(question))
    if not q_terms:
        return REFUSAL
    scored = []
    for i, c in enumerate(chunks, start=1):
        for sent in re.split(r"(?<=[.!?])\s+", c.text.replace("\n", " ")):
            terms = set(tokenize(sent))
            overlap = q_terms & terms
            if len(sent) < 30 or not overlap:
                continue
            # Long words tend to be the informative ones ("transformer" vs "use").
            score = sum(min(len(t), 10) for t in overlap) / (1 + 0.01 * len(sent))
            # Small bonus for higher-ranked chunks so ties go to the better source.
            scored.append((score * (1 - 0.05 * (i - 1)), i, sent.strip()))
    if not scored:
        return REFUSAL
    scored.sort(reverse=True)
    picked, used = [], set()
    for _, i, sent in scored:
        if sent in used:
            continue
        picked.append(f"{sent} [{i}]")
        used.add(sent)
        if len(picked) == max_sentences:
            break
    return " ".join(picked)


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------

def render_citations(raw: str, chunks: list[RetrievedChunk]) -> tuple[str, list[int], int, int]:
    """
    Replace [n] markers with [title, section]. Returns
    (rendered_text, valid_source_numbers, total_markers, valid_markers).
    Out-of-range numbers are dropped from the text rather than shown as broken links.
    """
    total = valid = 0
    cited: list[int] = []

    def swap(match: re.Match) -> str:
        nonlocal total, valid
        parts = []
        for num in re.split(r"[,;]", match.group(1)):
            n = int(num.strip())
            total += 1
            if 1 <= n <= len(chunks):
                valid += 1
                if n not in cited:
                    cited.append(n)
                c = chunks[n - 1]
                parts.append(f"[{c.title}, {c.section}]")
        return " ".join(parts)

    text = _CITE_RE.sub(swap, raw)
    # Tidy up spaces left behind by dropped markers, e.g. "word  ." -> "word."
    text = re.sub(r"\s+([.,;:])", r"\1", re.sub(r"[ \t]{2,}", " ", text)).strip()
    return text, cited, total, valid


def is_refusal(text: str) -> bool:
    t = text.lower().replace("’", "'")
    return "can't find this in the papers" in t or "cannot find this in the papers" in t


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def too_irrelevant(chunks: list[RetrievedChunk]) -> bool:
    """
    If the cross-encoder thinks even the best chunk is unrelated, don't let the
    LLM improvise - this is the cheapest hallucination guard there is.
    Note: look at the *highest* score, not chunks[0]. The final order blends the
    cross-encoder with the first-stage ranking, so the top chunk isn't always the
    one the cross-encoder liked most.
    """
    if not chunks:
        return True
    return chunks[0].score_type == "rerank" and max(c.score for c in chunks) < settings.min_rerank_score


def needs_repair(raw: str, chunks: list[RetrievedChunk]) -> bool:
    """An answer (not a refusal) that doesn't contain a single usable [n]."""
    return settings.cite_repair and not is_refusal(raw) and render_citations(raw, chunks)[3] == 0


def repair_citations(provider: str, model: str, user: str, raw: str) -> str:
    """One follow-up turn: "you forgot the source numbers - add them, or say you can't"."""
    return call_llm(provider, model, SYSTEM_PROMPT, user,
                    history=[{"role": "assistant", "content": raw}, {"role": "user", "content": REPAIR_PROMPT}])


def finish_answer(question: str, raw: str, chunks: list[RetrievedChunk], provider: str, model: str) -> Answer:
    """Turn the model's final text into an Answer: render citations, apply the refusal rules."""
    if is_refusal(raw):
        return Answer(question, REFUSAL, raw, True, [], 0, 0, provider, model, chunks, reason="model_declined")
    text, cited, total, valid = render_citations(raw, chunks)
    if valid == 0:
        # The model answered without pointing at any source -> treat as unsupported.
        return Answer(question, REFUSAL, raw, True, [], total, 0, provider, model, chunks, reason="no_citations")
    return Answer(question, text, raw, False, cited, total, valid, provider, model, chunks)


def generate_answer(question: str, chunks: list[RetrievedChunk], provider: str | None = None,
                    model: str | None = None) -> Answer:
    provider = provider or resolve_provider()
    model = model or model_for(provider)

    if too_irrelevant(chunks):
        return Answer(question, REFUSAL, REFUSAL, True, [], 0, 0, provider, model, chunks, reason="low_relevance")

    t0 = time.perf_counter()
    repaired = False
    if provider == "extractive":
        raw = extractive_answer(question, chunks)
    else:
        user = build_user_prompt(question, chunks)
        raw = call_llm(provider, model, SYSTEM_PROMPT, user)
        # An answer with no usable [n] would be refused. Give the model one
        # follow-up turn to cite it properly before giving up on it.
        if needs_repair(raw, chunks):
            raw = repair_citations(provider, model, user, raw)
            repaired = True
    gen_time = time.perf_counter() - t0

    ans = finish_answer(question, raw, chunks, provider, model)
    ans.timings["generation_s"] = gen_time
    ans.timings["cite_repair"] = repaired
    return ans


def chunk_to_dict(c: RetrievedChunk) -> dict:
    return {"chunk_id": c.chunk_id, "paper_id": c.paper_id, "title": c.title, "section": c.section,
            "text": c.text, "score": round(float(c.score), 4), "score_type": c.score_type}


def answer_events(question: str, retriever: Retriever, mode: str = "hybrid_rerank", k: int | None = None,
                  paper_id: str | None = None, provider: str | None = None,
                  model: str | None = None) -> Iterator[dict]:
    """
    The streaming version of answer_question(), for the web UI. Yields small dicts:

      {"type": "sources", "chunks": [...], "retrieval_s": ...}   as soon as retrieval is done
      {"type": "token", "text": "..."}                           while the model writes
      {"type": "repair"}                                         if the citation repair turn runs
      {"type": "final", "answer": {...}}                         the finished, checked answer

    It goes through the same too_irrelevant / needs_repair / finish_answer steps as
    generate_answer(), so what the UI shows is what evaluate.py measured.
    """
    provider = provider or resolve_provider()
    model = model or model_for(provider)

    t0 = time.perf_counter()
    chunks = retriever.retrieve(question, mode=mode, k=k, paper_id=paper_id)
    retrieval_s = time.perf_counter() - t0
    yield {"type": "sources", "chunks": [chunk_to_dict(c) for c in chunks], "retrieval_s": retrieval_s}

    t1 = time.perf_counter()
    repaired = False
    if too_irrelevant(chunks):
        ans = Answer(question, REFUSAL, REFUSAL, True, [], 0, 0, provider, model, chunks, reason="low_relevance")
    else:
        if provider == "extractive":
            raw = extractive_answer(question, chunks)
            yield {"type": "token", "text": raw}
        else:
            user = build_user_prompt(question, chunks)
            parts = []
            for piece in stream_llm(provider, model, SYSTEM_PROMPT, user):
                parts.append(piece)
                yield {"type": "token", "text": piece}
            raw = "".join(parts).strip()
            if needs_repair(raw, chunks):
                yield {"type": "repair"}
                raw = repair_citations(provider, model, user, raw)
                repaired = True
        ans = finish_answer(question, raw, chunks, provider, model)

    yield {"type": "final", "answer": {
        "text": ans.text, "raw": ans.raw, "refused": ans.refused, "reason": ans.reason,
        "cited": ans.cited, "citation_accuracy": ans.citation_accuracy,
        "provider": ans.provider, "model": ans.model, "cite_repair": repaired,
        "retrieval_s": retrieval_s, "generation_s": time.perf_counter() - t1,
    }}


def answer_question(question: str, retriever: Retriever, mode: str = "hybrid_rerank", k: int | None = None,
                    paper_id: str | None = None, provider: str | None = None) -> Answer:
    """Retrieve + generate in one go. This is what the app and the CLI call."""
    t0 = time.perf_counter()
    chunks = retriever.retrieve(question, mode=mode, k=k, paper_id=paper_id)
    retrieval_time = time.perf_counter() - t0
    ans = generate_answer(question, chunks, provider=provider)
    ans.timings["retrieval_s"] = retrieval_time
    return ans


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the papers a question from the terminal.")
    parser.add_argument("question")
    parser.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    parser.add_argument("--provider", default=None, help="ollama | openai | anthropic | extractive")
    args = parser.parse_args()

    retriever = Retriever(load_reranker=args.mode == "hybrid_rerank")
    ans = answer_question(args.question, retriever, mode=args.mode, provider=args.provider)
    print(f"\n{ans.text}\n")
    print(f"(provider={ans.provider} model={ans.model or '-'} "
          f"retrieval={ans.timings.get('retrieval_s', 0):.2f}s "
          f"generation={ans.timings.get('generation_s', 0):.2f}s)")
    for i, c in enumerate(ans.chunks, start=1):
        print(f"  [{i}] {c.title} :: {c.section}  ({c.score_type}={c.score:.3f})")


if __name__ == "__main__":
    main()
