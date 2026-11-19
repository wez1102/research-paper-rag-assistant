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
from dataclasses import dataclass, field
from typing import Iterator

import httpx

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

    # v3 targets what went wrong with v2 on the round-2 dev answer set (dev papers
    # only; see docs/EVALUATION.md):
    #  * multi-paper questions: facts from different papers were merged into one list
    #    with no word on which paper said what, and unrelated papers crept in;
    #  * "By how much...?" style questions with no number in the sources got a vague
    #    on-topic sentence instead of a refusal;
    #  * the papers' own reference markers (BIBREF3, TABREF1) were copied into answers.
    "v3": f"""You answer questions about NLP research papers using ONLY the numbered sources provided.

How to answer:
1. Start with the shortest direct answer: a name, a number, a short list, or "Yes"/"No" for a yes/no question. Use the exact terms the sources use.
2. Optionally add one or two short sentences of supporting detail.
3. End every sentence that states a fact with the number(s) of the source(s) that say it, like [1] or [2][4].

Rules:
- Use only the sources. Do not use outside knowledge.
- Sources often use different words than the question (synonyms, abbreviations, paraphrases). If a source answers the question in other words, that counts - answer it.
- Answer the exact question. If the sources are about the topic but do not contain what is asked (for example the question asks "how much" and the sources give no number), reply with exactly: {REFUSAL}
- The sources may come from different papers. Use only the papers that answer the question. When your answer uses more than one paper, say which paper each fact comes from (a short form of its title is enough).
- Do not copy the papers' reference markers such as BIBREF3, TABREF1 or FIGREF2.
- If none of the sources is relevant to the question, reply with exactly: {REFUSAL}""",
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
_PAPER_MARKER_RE = re.compile(r"\s*\b(?:BIBREF|TABREF|FIGREF|INLINEFORM|DISPLAYFORM|SECREF)\d+\b")


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
    # "low_relevance" (reranker floor), "model_declined", "no_citations",
    # "needs_paper" (corpus-wide question that doesn't say which paper), or "".
    reason: str = ""
    # For "needs_paper": the papers the question most likely means, best first.
    candidates: list = field(default_factory=list)

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

class LLMError(RuntimeError):
    """Something went wrong talking to the LLM. `code` is a short machine-readable reason."""
    code = "llm_error"


class LLMUnavailable(LLMError):
    """Couldn't connect at all (Ollama not running, wrong host, network down)."""
    code = "llm_unavailable"


class LLMTimeout(LLMError):
    """Connected, but the model took too long to answer (or stopped mid-answer)."""
    code = "llm_timeout"


class LLMBadResponse(LLMError):
    """The server answered with something that isn't what the API promises."""
    code = "llm_bad_response"


def _timeout() -> httpx.Timeout:
    # Connect fails fast (is anything listening?); read is per piece of the answer,
    # and has to allow for the model loading into memory on the first request.
    return httpx.Timeout(connect=settings.llm_connect_timeout, read=settings.llm_read_timeout,
                         write=settings.llm_connect_timeout, pool=settings.llm_connect_timeout)


def _ollama_up() -> bool:
    try:
        httpx.get(f"{settings.ollama_host}/api/tags", timeout=1.0).raise_for_status()
        return True
    except Exception:
        return False


def ollama_models() -> list[dict]:
    """
    The chat models installed in the local Ollama, for the model picker in the UI.
    Raises LLMUnavailable if Ollama isn't reachable, so the caller can tell "not
    running" apart from "running but nothing pulled yet".
    """
    try:
        resp = httpx.get(f"{settings.ollama_host}/api/tags", timeout=2.0)
        resp.raise_for_status()
        tags = resp.json().get("models", [])
    except httpx.HTTPError as err:
        raise LLMUnavailable(f"Ollama isn't reachable at {settings.ollama_host}: {err}") from err
    except ValueError as err:
        raise LLMBadResponse(f"Ollama returned something that isn't JSON: {err}") from err
    models = []
    for m in tags:
        details = m.get("details") or {}
        family = (details.get("family") or "").lower()
        # Embedding models (nomic-embed-text, bge-m3...) show up here too but can't chat.
        if "embed" in m.get("name", "").lower() or family in {"bert", "nomic-bert"}:
            continue
        models.append({
            "name": m["name"],
            "size_gb": round(m.get("size", 0) / 1e9, 1),
            "params": details.get("parameter_size", ""),
            "quant": details.get("quantization_level", ""),
        })
    return sorted(models, key=lambda m: m["name"])


def _ollama_payload(model: str, system: str, turns: list[dict], stream: bool) -> dict:
    return {
        "model": model,
        "stream": stream,
        # Reasoning models (Qwen3, DeepSeek-R1...) would otherwise "think out loud"
        # first, which is slow and pollutes the answer. Plain models ignore this.
        "think": False,
        "messages": [{"role": "system", "content": system}] + turns,
        "options": {"temperature": 0, "seed": settings.seed, "num_predict": settings.max_answer_tokens},
    }


def _ollama_chat(model: str, system: str, turns: list[dict]) -> str:
    """
    One non-streaming chat call. A refused connection is retried once after a short
    pause (Ollama restarting, or still binding its port); a timeout is not retried,
    because the model was clearly busy and a second try would just double the wait.
    """
    url = f"{settings.ollama_host}/api/chat"
    payload = _ollama_payload(model, system, turns, stream=False)
    for attempt in (1, 2):
        try:
            resp = httpx.post(url, json=payload, timeout=_timeout())
            break
        except httpx.ConnectError as err:
            if attempt == 2:
                raise LLMUnavailable(f"Couldn't reach Ollama at {settings.ollama_host}") from err
            time.sleep(1.0)
        except httpx.TimeoutException as err:
            raise LLMTimeout(f"Ollama didn't answer within {settings.llm_read_timeout:.0f}s") from err
    if resp.status_code != 200:
        raise LLMBadResponse(f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        content = resp.json()["message"]["content"]
    except (ValueError, KeyError, TypeError) as err:
        raise LLMBadResponse(f"Unexpected reply from Ollama: {resp.text[:200]}") from err
    # Older Ollama versions ignore "think" and inline the reasoning instead.
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def _ollama_stream(model: str, system: str, turns: list[dict]) -> Iterator[str]:
    """
    Stream one chat from Ollama, which sends one JSON object per line. Guards:
      * a line that isn't valid JSON, or an {"error": ...} line -> LLMBadResponse
      * the connection closing before Ollama's final {"done": true} -> LLMTimeout
        (the answer was cut off, so it must not be shown as if it were complete)
      * the whole answer taking longer than llm_total_timeout -> LLMTimeout
    If the caller stops iterating (the browser went away), the `with` block closes
    the HTTP connection, and Ollama stops generating for us.
    """
    deadline = time.monotonic() + settings.llm_total_timeout
    try:
        with httpx.stream("POST", f"{settings.ollama_host}/api/chat",
                          json=_ollama_payload(model, system, turns, stream=True), timeout=_timeout()) as resp:
            if resp.status_code != 200:
                resp.read()
                raise LLMBadResponse(f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}")
            done = False
            for line in resp.iter_lines():
                # Checked on every line: a model that is slow throughout still gets cut
                # off, even if each piece arrives inside the per-read timeout.
                if time.monotonic() > deadline:
                    raise LLMTimeout(f"The answer took longer than {settings.llm_total_timeout:.0f}s")
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line)
                except ValueError as err:
                    raise LLMBadResponse(f"Ollama sent a malformed line: {line[:120]!r}") from err
                if msg.get("error"):
                    raise LLMBadResponse(str(msg["error"]))
                piece = (msg.get("message") or {}).get("content", "")
                if piece:
                    yield piece
                if msg.get("done"):
                    done = True
                    break
            if not done:
                raise LLMTimeout("Ollama closed the stream before the answer was finished")
    except httpx.ConnectError as err:
        raise LLMUnavailable(f"Couldn't reach Ollama at {settings.ollama_host}") from err
    except httpx.TimeoutException as err:
        raise LLMTimeout(f"Ollama went quiet for more than {settings.llm_read_timeout:.0f}s") from err
    except httpx.RemoteProtocolError as err:
        raise LLMTimeout("The connection to Ollama dropped mid-answer") from err


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
        return _ollama_chat(model, system, turns)

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
        yield from _strip_think(_ollama_stream(model, system, [{"role": "user", "content": user}]))
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
    # The papers' own reference markers ("RoBERTa BIBREF2") mean nothing to a reader.
    text = _PAPER_MARKER_RE.sub("", text)
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


# Questions that are explicitly about several papers ("Which papers use ELMo?") are
# fine without a paper filter, so they never get the "which paper?" reply.
_CROSS_PAPER = re.compile(r"\b(papers|each paper|which paper|these works|across (?:the |these |all )?papers)\b", re.I)
NEEDS_PAPER = "This question doesn't say which paper it's about, and several papers could fit."


def paper_candidates(question: str, chunks: list[RetrievedChunk], retriever, paper_id: str | None) -> list[dict]:
    """
    Non-empty when we should ask "which paper?" instead of answering: no paper was
    picked, the question isn't explicitly cross-paper, and none of its words is
    specific to a few papers. Returns up to three likely papers, best first.

    Why: on the dev split, for such generic questions the top-ranked passage came
    from the asker's paper only 19% of the time, so an answer would usually describe
    the wrong paper - confidently, and with real citations.
    """
    if (not settings.clarify_generic or paper_id or not chunks or _CROSS_PAPER.search(question)
            or not hasattr(retriever, "question_specificity")):
        return []
    if retriever.question_specificity(question)["specific"]:
        return []
    out, seen = [], set()
    for c in chunks:
        if c.paper_id not in seen:
            seen.add(c.paper_id)
            out.append({"paper_id": c.paper_id, "title": c.title})
    return out[:3]


def clarify_answer(question: str, chunks: list[RetrievedChunk], candidates: list[dict], provider: str,
                   model: str) -> Answer:
    text = NEEDS_PAPER + " Pick one, for example: " + "; ".join(c["title"] for c in candidates) + "."
    return Answer(question, text, text, True, [], 0, 0, provider, model, chunks, reason="needs_paper",
                  candidates=candidates)


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
    timings: dict = {}
    chunks = retriever.retrieve(question, mode=mode, k=k, paper_id=paper_id, timings=timings)
    retrieval_s = time.perf_counter() - t0
    yield {"type": "sources", "chunks": [chunk_to_dict(c) for c in chunks], "retrieval_s": retrieval_s,
           "timings": {k_: v for k_, v in timings.items() if k_.endswith("_s")}}

    t1 = time.perf_counter()
    repaired = False
    candidates = [] if too_irrelevant(chunks) else paper_candidates(question, chunks, retriever, paper_id)
    if too_irrelevant(chunks):
        ans = Answer(question, REFUSAL, REFUSAL, True, [], 0, 0, provider, model, chunks, reason="low_relevance")
    elif candidates:
        ans = clarify_answer(question, chunks, candidates, provider, model)
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
        "cited": ans.cited, "citation_accuracy": ans.citation_accuracy, "candidates": ans.candidates,
        "provider": ans.provider, "model": ans.model, "cite_repair": repaired,
        "retrieval_s": retrieval_s, "generation_s": time.perf_counter() - t1,
    }}


def answer_question(question: str, retriever: Retriever, mode: str = "hybrid_rerank", k: int | None = None,
                    paper_id: str | None = None, provider: str | None = None) -> Answer:
    """Retrieve + generate in one go. This is what the app and the CLI call."""
    t0 = time.perf_counter()
    chunks = retriever.retrieve(question, mode=mode, k=k, paper_id=paper_id)
    retrieval_time = time.perf_counter() - t0
    candidates = [] if too_irrelevant(chunks) else paper_candidates(question, chunks, retriever, paper_id)
    if candidates:
        provider = provider or resolve_provider()
        ans = clarify_answer(question, chunks, candidates, provider, model_for(provider))
    else:
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
