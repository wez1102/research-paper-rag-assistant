"""
Round-2 answer evaluation: is the answer right, and does each cited passage
actually back up what the answer says?

The round-1 "citation accuracy" only checked that every [n] pointed at one of the
retrieved passages - a model citing [1] for a sentence passage [1] never mentions
still scored 100%. This script measures the things a reader cares about:

  answer correctness   judged against QASPER's reference answers (or our own, for
                       the questions we wrote)
  claim support        each sentence of the answer is a claim; is it supported by
                       the passages it cites?
  citation coverage    share of claims that cite anything at all
  refusal behaviour    refuses the unanswerable / off-topic ones, answers the rest
  ambiguity handling   for questions that don't say which paper they mean, does the
                       answer say which paper(s) it is talking about?

The questions live in eval/answer_set_{dev,final}.jsonl (see docs/EVALUATION.md
for how they were picked and reviewed). Stages:

  python answer_eval.py retrieve --set eval/answer_set_dev.jsonl --out results/v2/answers/dev
      Runs the real Retriever per question (the right split's index, the right scope),
      applies the relevance floor, and writes the exact prompts the LLM would get.
  python answer_eval.py generate --out results/v2/answers/dev --provider ollama
      Native path: asks the LLM through generate.py (needs Ollama/API reachable).
  python answer_eval.py ingest --out ... --outputs relay_outputs.json
      Relay path: model outputs produced elsewhere (in this project's round-2 run,
      by a browser tab next to Ollama on the Mac). Every prompt is SHA-1 checked
      against the one built here, so they are provably the same prompts.
  python answer_eval.py judge-jobs --out ...      writes LLM-judge prompts
  python answer_eval.py score --out ... [--review eval/review_final.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

import pandas as pd

from config import settings
from generate import (PROMPTS, REFUSAL, REPAIR_PROMPT, build_user_prompt, clarify_answer, finish_answer,
                      generate_answer, needs_repair, paper_candidates, too_irrelevant)
from retrieve import RetrievedChunk, Retriever

_CITE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


# ---------------------------------------------------------------------------
# Helpers shared by every stage
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def messages_sha1(messages: list[dict]) -> str:
    """Same bytes as JSON.stringify(messages) in the browser (compact, unicode kept)."""
    return hashlib.sha1(json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def chunk_block(c: RetrievedChunk | dict) -> str:
    """The text build_user_prompt() puts after "[n] " for one source."""
    get = (lambda k: c[k]) if isinstance(c, dict) else (lambda k: getattr(c, k))
    return f"Paper: {get('title')}\nSection: {get('section')}\n{get('text')}"


def to_chunk(d: dict) -> RetrievedChunk:
    return RetrievedChunk(d["chunk_id"], d["paper_id"], d["title"], d["section"], d["text"],
                          d.get("paragraph_ids", []), d["score"], d["score_type"])


def split_claims(raw: str) -> list[dict]:
    """
    Cut an answer into sentence-level claims and the source numbers each one cites.
    A citation that stands after the full stop ("... BLEU. [2]") belongs to the
    sentence before it, which is how small models often write them.
    """
    text = raw.strip()
    # Move "[n]" that follows a sentence end back in front of the full stop.
    text = re.sub(r"([.!?])\s*((?:\[\d+(?:\s*[,;]\s*\d+)*\]\s*)+)", lambda m: " " + m.group(2).strip() + m.group(1) + " ", text)
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+", text) if p.strip()]
    claims = []
    for p in parts:
        nums = sorted({int(n) for m in _CITE.finditer(p) for n in re.split(r"[,;]", m.group(1))})
        body = _CITE.sub("", p).strip(" .;:-")
        if len(body) < 2:
            continue
        claims.append({"text": body, "cites": nums})
    return claims


# ---------------------------------------------------------------------------
# Stage 1: retrieval + prompts
# ---------------------------------------------------------------------------

def stage_retrieve(args) -> None:
    items = read_jsonl(Path(args.set))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    system = PROMPTS[settings.prompt_version]
    retrievers: dict[str, Retriever] = {}
    rows, jobs, chunk_store = [], [], {}
    for it in items:
        split = it["split"]
        if split not in retrievers:
            retrievers[split] = Retriever(collection_name=settings.collection_name(split=split))
        r = retrievers[split]
        timings: dict = {}
        chunks = r.retrieve(it["question"], mode="hybrid_rerank", paper_id=it.get("paper_id"), timings=timings)
        floor = too_irrelevant(chunks)
        # Same order as the app: relevance floor first, then "which paper?".
        candidates = [] if floor else paper_candidates(it["question"], chunks, r, it.get("paper_id"))
        row = {**it, "prompt_version": settings.prompt_version, "floor_refused": floor,
               "candidates": candidates,
               "max_rerank": max((c.score for c in chunks), default=None), "timings": timings,
               "chunks": [{"chunk_id": c.chunk_id, "paper_id": c.paper_id, "title": c.title, "section": c.section,
                           "text": c.text, "paragraph_ids": c.paragraph_ids, "score": c.score,
                           "score_type": c.score_type} for c in chunks]}
        rows.append(row)
        if floor or candidates:
            continue
        user = build_user_prompt(it["question"], chunks)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        # The relay copy swaps each source block for a placeholder, so a passage used by
        # ten questions is sent once. The browser puts them back and checks the SHA-1.
        templ = user
        for i, c in enumerate(chunks, start=1):
            chunk_store[c.chunk_id] = chunk_block(c)
            templ = templ.replace(f"[{i}] {chunk_block(c)}", f"[{i}] {{{{C:{c.chunk_id}}}}}", 1)
        jobs.append({"id": it["id"], "kind": "answer",
                     "messages": [{"role": "system", "content": "{{SYS:" + settings.prompt_version + "}}"},
                                  {"role": "user", "content": templ}],
                     "sha1": messages_sha1(messages)})
    write_jsonl(out / "retrieval.jsonl", rows)
    json.dump({"systems": {settings.prompt_version: system}, "chunks": chunk_store, "jobs": jobs},
              (out / "relay_answer_jobs.json").open("w"), ensure_ascii=False)
    n_floor = sum(r["floor_refused"] for r in rows)
    n_clar = sum(bool(r["candidates"]) for r in rows)
    print(f"{len(rows)} questions, {n_floor} refused by the relevance floor, {n_clar} asked 'which paper?', "
          f"{len(jobs)} LLM jobs, "
          f"{len(chunk_store)} distinct passages -> {out}")


def repair_jobs(out: Path, answers: dict[str, dict]) -> list[dict]:
    """Second-turn jobs for answers that cited nothing usable (same rule as generate.needs_repair)."""
    rows = {r["id"]: r for r in read_jsonl(out / "retrieval.jsonl")}
    bundle = json.load((out / "relay_answer_jobs.json").open())
    by_id = {j["id"]: j for j in bundle["jobs"]}
    jobs = []
    for qid, a in answers.items():
        chunks = [to_chunk(c) for c in rows[qid]["chunks"]]
        if a.get("repair_of") or not needs_repair(a["content"], chunks):
            continue
        base = by_id[qid]["messages"]
        msgs = base + [{"role": "assistant", "content": a["content"]}, {"role": "user", "content": REPAIR_PROMPT}]
        system = bundle["systems"][rows[qid]["prompt_version"]]
        full = [{"role": "system", "content": system},
                {"role": "user", "content": build_user_prompt(rows[qid]["question"], chunks)},
                {"role": "assistant", "content": a["content"]}, {"role": "user", "content": REPAIR_PROMPT}]
        jobs.append({"id": qid + "#repair", "kind": "repair", "messages": msgs, "sha1": messages_sha1(full)})
    return jobs


# ---------------------------------------------------------------------------
# Stage 2: answers (native, or ingested from the relay)
# ---------------------------------------------------------------------------

def stage_generate(args) -> None:
    """Native path: the same generate_answer() the app uses, incl. the repair turn."""
    out = Path(args.out)
    answers = []
    for row in read_jsonl(out / "retrieval.jsonl"):
        chunks = [to_chunk(c) for c in row["chunks"]]
        if row.get("candidates"):
            ans = clarify_answer(row["question"], chunks, row["candidates"], args.provider or "", args.model or "")
        else:
            ans = generate_answer(row["question"], chunks, provider=args.provider, model=args.model)
        answers.append({"id": row["id"], "raw": ans.raw, "refused": ans.refused, "reason": ans.reason,
                        "cite_repair": ans.timings.get("cite_repair", False), "provider": ans.provider,
                        "model": ans.model, "generation_s": ans.timings.get("generation_s")})
        print(f"{row['id']}: {'REFUSED ' + ans.reason if ans.refused else ans.raw[:80]}")
    write_jsonl(out / "answers.jsonl", answers)


def stage_ingest(args) -> None:
    """
    Relay path. `outputs` maps job id -> {"content", "sha1", timings...} as produced by
    the browser runner. Prompts whose SHA-1 doesn't match are rejected.
    """
    out = Path(args.out)
    outputs = json.load(open(args.outputs))
    bundle = json.load((out / "relay_answer_jobs.json").open())
    expected = {j["id"]: j["sha1"] for j in bundle["jobs"]}
    if (out / "relay_repair_jobs.json").exists():
        expected.update({j["id"]: j["sha1"] for j in json.load((out / "relay_repair_jobs.json").open())["jobs"]})
    bad = [k for k, v in outputs.items() if k in expected and v["sha1"] != expected[k]]
    if bad:
        raise SystemExit(f"SHA-1 mismatch for {bad} - those prompts differ from the ones built here.")
    missing = [k for k in expected if k not in outputs and not k.endswith("#repair")]
    print(f"{len(outputs)} outputs, all prompts verified; {len(missing)} answer jobs still missing")

    first = {k: v for k, v in outputs.items() if not k.endswith("#repair")}
    todo = repair_jobs(out, first)
    if todo and not all(j["id"] in outputs for j in todo):
        json.dump({"systems": bundle["systems"], "chunks": {}, "jobs": todo},
                  (out / "relay_repair_jobs.json").open("w"), ensure_ascii=False)
        print(f"{len(todo)} answers need the citation-repair turn -> relay_repair_jobs.json (run those, then ingest again)")
        return

    rows = read_jsonl(out / "retrieval.jsonl")
    answers = []
    for row in rows:
        if row["floor_refused"]:
            answers.append({"id": row["id"], "raw": REFUSAL, "refused": True, "reason": "low_relevance",
                            "cite_repair": False, "generation_s": 0.0})
            continue
        if row.get("candidates"):
            ans = clarify_answer(row["question"], [], row["candidates"], "ollama", args.model)
            answers.append({"id": row["id"], "raw": ans.raw, "refused": True, "reason": "needs_paper",
                            "candidates": row["candidates"], "cite_repair": False, "generation_s": 0.0})
            continue
        o = outputs.get(row["id"])
        if o is None:
            continue
        raw, repaired = o["content"].strip(), False
        if row["id"] + "#repair" in outputs:
            raw, repaired = outputs[row["id"] + "#repair"]["content"].strip(), True
        chunks = [to_chunk(c) for c in row["chunks"]]
        ans = finish_answer(row["question"], raw, chunks, "ollama", args.model)
        answers.append({"id": row["id"], "raw": ans.raw, "text": ans.text, "refused": ans.refused,
                        "reason": ans.reason, "cite_repair": repaired, "model": args.model,
                        "generation_s": o.get("total_s"), "ttft_s": o.get("ttft_s"),
                        "eval_tokens": o.get("eval_count"), "prompt_tokens": o.get("prompt_eval_count")})
    write_jsonl(out / "answers.jsonl", answers)
    print(f"wrote {len(answers)} answers -> {out / 'answers.jsonl'}")


# ---------------------------------------------------------------------------
# Stage 3: judge prompts
# ---------------------------------------------------------------------------

JUDGE_SUPPORT = """You check citations in answers about research papers.

Passage(s):
{passages}

Claim: {claim}

Does the passage text above state or directly imply the claim? Judge only from the passage, not from what you know.
Reply with one word - SUPPORTED, PARTIAL (some of the claim is in the passage, some is not), or UNSUPPORTED - then " | " and a reason of at most 15 words."""

JUDGE_CORRECT = """You grade answers to questions about a research paper.

Question: {question}
Reference answer(s) written by people who read the paper (any one of them counts as right):
{references}

System answer: {answer}

Is the system answer correct? Ignore citation markers, wording and extra detail that does not contradict the references.
Reply with one word - CORRECT, PARTIAL (right but incomplete, or partly wrong), or INCORRECT - then " | " and a reason of at most 15 words."""


def stage_judge_jobs(args) -> None:
    out = Path(args.out)
    rows = {r["id"]: r for r in read_jsonl(out / "retrieval.jsonl")}
    answers = read_jsonl(out / "answers.jsonl")
    jobs, store = [], {}
    for a in answers:
        row = rows[a["id"]]
        if a["refused"]:
            continue
        if row.get("references"):
            refs = "\n".join(f"- {r}" for r in row["references"])
            plain = _CITE.sub("", a["raw"]).strip()
            prompt = JUDGE_CORRECT.format(question=row["question"], references=refs, answer=plain)
            jobs.append({"id": f"{a['id']}#correct", "kind": "correct", "model": args.judge_model,
                         "messages": [{"role": "user", "content": prompt}],
                         "sha1": messages_sha1([{"role": "user", "content": prompt}])})
        for ci, claim in enumerate(split_claims(a["raw"])):
            valid = [n for n in claim["cites"] if 1 <= n <= len(row["chunks"])]
            if not valid:
                continue
            claim_text = claim["text"]
            if len(claim_text.split()) < 4:
                # "No" or "100" means nothing without the question it answers.
                claim_text = f'The answer to "{row["question"]}" is: {claim_text}'
            claim = {**claim, "text": claim_text}
            full = "\n\n".join(f"[{n}] {chunk_block(row['chunks'][n - 1])}" for n in valid)
            templ = "\n\n".join(f"[{n}] {{{{C:{row['chunks'][n - 1]['chunk_id']}}}}}" for n in valid)
            for n in valid:
                store[row["chunks"][n - 1]["chunk_id"]] = chunk_block(row["chunks"][n - 1])
            prompt = JUDGE_SUPPORT.format(passages=full, claim=claim["text"])
            jobs.append({"id": f"{a['id']}#claim{ci}", "kind": "support", "model": args.judge_model,
                         "messages": [{"role": "user", "content": JUDGE_SUPPORT.format(passages=templ,
                                                                                     claim=claim["text"])}],
                         "sha1": messages_sha1([{"role": "user", "content": prompt}])})
    json.dump({"systems": {}, "chunks": store, "jobs": jobs}, (out / "relay_judge_jobs.json").open("w"),
              ensure_ascii=False)
    print(f"{len(jobs)} judge jobs -> {out / 'relay_judge_jobs.json'}")


# ---------------------------------------------------------------------------
# Stage 3b: a sheet for manual review
# ---------------------------------------------------------------------------

def stage_review_sheet(args) -> None:
    """
    Writes review_sheet.md (every answer with its references, claims and the cited
    passages) and review_template.json (the keys to fill in). Labels:
      <id>#correct   CORRECT | PARTIAL | INCORRECT          (answerable, with references)
      <id>#claimN    SUPPORTED | PARTIAL | UNSUPPORTED      (each cited claim)
      <id>#behaviour OK | FAIL                              (unanswerable/off-topic/ambiguous/multi-paper:
                                                             refused or clarified when it should, or for
                                                             multi-paper: right papers, attributed)
    """
    out = Path(args.out)
    rows = {r["id"]: r for r in read_jsonl(out / "retrieval.jsonl")}
    answers = read_jsonl(out / "answers.jsonl")
    lines, template = [f"# Review sheet: {out}\n"], {}
    for a in answers:
        row = rows[a["id"]]
        lines.append(f"## {a['id']} ({row['category']}, expected: {row['expected']})\n")
        lines.append(f"**Q:** {row['question']}  \n**Paper:** {row.get('title') or '(none picked)'}\n")
        for ref in row.get("references", []):
            lines.append(f"- REF: {ref}")
        lines.append(f"\n**Answer** ({a.get('reason') or 'answered'}): {a['raw']}\n")
        if row["category"] != "answerable" or not row.get("references"):
            template[f"{a['id']}#behaviour"] = ""
        if not a["refused"]:
            if row.get("references"):
                template[f"{a['id']}#correct"] = ""
            for ci, c in enumerate(split_claims(a["raw"])):
                lines.append(f"- claim{ci} cites {c['cites']}: {c['text']}")
                valid = [n for n in c["cites"] if 1 <= n <= len(row["chunks"])]
                if valid:
                    template[f"{a['id']}#claim{ci}"] = ""
                for n in valid:
                    ch = row["chunks"][n - 1]
                    lines.append(f"  - [{n}] {ch['title'][:60]} / {ch['section']}: {ch['text']}")
        lines.append("")
    (out / "review_sheet.md").write_text("\n".join(lines))
    (out / "review_template.json").write_text(json.dumps(template, indent=1))
    print(f"{len(template)} labels to fill -> {out / 'review_template.json'}")


# ---------------------------------------------------------------------------
# Stage 4: scores
# ---------------------------------------------------------------------------

def _label(text: str, allowed: tuple[str, ...]) -> str:
    head = text.strip().split("|")[0].strip().upper()
    for a in allowed:
        if head.startswith(a):
            return a
    return "UNPARSED"


def stage_score(args) -> None:
    out = Path(args.out)
    rows = {r["id"]: r for r in read_jsonl(out / "retrieval.jsonl")}
    answers = {a["id"]: a for a in read_jsonl(out / "answers.jsonl")}
    judge = json.load(open(args.judgments)) if args.judgments else {}
    review = json.load(open(args.review)) if args.review else {}
    per = []
    for qid, a in answers.items():
        row = rows[qid]
        cat, expect = row["category"], row["expected"]
        claims = [] if a["refused"] else split_claims(a["raw"])
        sup = []
        for ci, c in enumerate(claims):
            key = f"{qid}#claim{ci}"
            human = review.get(key)
            j = _label(judge[key]["content"], ("SUPPORTED", "PARTIAL", "UNSUPPORTED")) if key in judge else None
            sup.append({"claim": c["text"], "cites": c["cites"], "judge": j, "review": human})
        corr_j = _label(judge[f"{qid}#correct"]["content"], ("CORRECT", "PARTIAL", "INCORRECT")) \
            if f"{qid}#correct" in judge else None
        corr_r = review.get(f"{qid}#correct")
        if expect == "answer" and a["refused"] and row.get("references"):
            # A refused answerable question is a miss, not a gap in the data.
            corr_j = corr_r = "REFUSED"
        paper_named = None
        if cat in ("ambiguous", "multi_paper") and not a["refused"]:
            # Does the answer say which paper each fact is from? A reviewer's call.
            paper_named = review.get(f"{qid}#attributed")
        per.append({
            "id": qid, "category": cat, "expected": expect, "question": row["question"],
            "refused": a["refused"], "reason": a.get("reason", ""), "cite_repair": a.get("cite_repair", False),
            "answer": a["raw"], "n_claims": len(claims),
            "n_claims_cited": sum(1 for c in claims if c["cites"]),
            "support_judge": [s["judge"] for s in sup], "support_review": [s["review"] for s in sup],
            "correct_judge": corr_j, "correct_review": corr_r, "attributed_review": paper_named,
            "behaviour_review": review.get(f"{qid}#behaviour"),
            "generation_s": a.get("generation_s"), "ttft_s": a.get("ttft_s"),
        })
    df = pd.DataFrame(per)
    df.to_csv(out / "answer_eval_per_question.csv", index=False)
    summary = summarize_answers(df)
    summary["judge_vs_review"] = judge_agreement(df)
    (out / "answer_eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def _rate(values, good: set) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(v in good for v in vals) / len(vals), 3) if vals else None


def judge_agreement(df: pd.DataFrame) -> dict:
    """How often the automated judge agrees with the reviewer, where both labelled the same item."""
    pairs_c = [(j, r) for j, r in zip(df["correct_judge"], df["correct_review"])
               if j not in (None, "REFUSED") and r not in (None, "REFUSED") and j == j and r == r]
    pairs_s = [(j, r) for js, rs in zip(df["support_judge"], df["support_review"]) for j, r in zip(js, rs)
               if j is not None and r is not None]
    def rate(pairs, f=lambda x: x):
        return round(sum(f(a) == f(b) for a, b in pairs) / len(pairs), 3) if pairs else None
    supported = lambda x: x == "SUPPORTED"                       # noqa: E731
    return {"correctness_n": len(pairs_c), "correctness_exact": rate(pairs_c),
            "correctness_strict_binary": rate(pairs_c, lambda x: x == "CORRECT"),
            "claims_n": len(pairs_s), "claims_exact": rate(pairs_s), "claims_binary": rate(pairs_s, supported),
            "judge_says_supported_reviewer_not": sum(1 for j, r in pairs_s if j == "SUPPORTED" and r != "SUPPORTED"),
            "reviewer_says_supported_judge_not": sum(1 for j, r in pairs_s if r == "SUPPORTED" and j != "SUPPORTED")}


def summarize_answers(df: pd.DataFrame) -> dict:
    s: dict = {"n": len(df)}
    ans = df[df["expected"] == "answer"]
    s["answerable_n"] = len(ans)
    s["answerable_refused"] = int(ans["refused"].sum())
    for who in ("review", "judge"):
        s[f"correct_{who}"] = _rate(ans[f"correct_{who}"], {"CORRECT"})
        s[f"correct_or_partial_{who}"] = _rate(ans[f"correct_{who}"], {"CORRECT", "PARTIAL"})
        flat = [x for lst in df[f"support_{who}"] for x in lst]
        s[f"claims_supported_{who}"] = _rate(flat, {"SUPPORTED"})
        s[f"claims_supported_or_partial_{who}"] = _rate(flat, {"SUPPORTED", "PARTIAL"})
        s[f"claims_judged_{who}"] = sum(1 for x in flat if x is not None)
    total, cited = int(df["n_claims"].sum()), int(df["n_claims_cited"].sum())
    s["citation_coverage"] = round(cited / total, 3) if total else None
    s["n_claims"] = total
    for cat in ("unanswerable", "off_topic"):
        sub = df[df["category"] == cat]
        if len(sub):
            s[f"{cat}_refused"] = f"{int(sub['refused'].sum())}/{len(sub)}"
    for cat in ("unanswerable", "off_topic", "ambiguous", "multi_paper"):
        sub = df[df["category"] == cat]
        if len(sub) and sub["behaviour_review"].notna().any():
            s[f"{cat}_ok_review"] = f"{int((sub['behaviour_review'] == 'OK').sum())}/{len(sub)}"
    gen = [g for g in df["generation_s"] if g is not None and g == g and g > 0]
    if gen:
        s["generation_p50_s"] = round(statistics.median(gen), 2)
        s["generation_p95_s"] = round(float(pd.Series(gen).quantile(0.95)), 2)
    ttft = [t for t in df["ttft_s"] if t is not None and t == t]
    if ttft:
        s["ttft_p50_s"] = round(statistics.median(ttft), 2)
        s["ttft_p95_s"] = round(float(pd.Series(ttft).quantile(0.95)), 2)
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("retrieve"); p.add_argument("--set", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("generate"); p.add_argument("--out", required=True)
    p.add_argument("--provider", default=None); p.add_argument("--model", default=None)
    p = sub.add_parser("ingest"); p.add_argument("--out", required=True); p.add_argument("--outputs", required=True)
    p.add_argument("--model", default="qwen3:8b")
    p = sub.add_parser("judge-jobs"); p.add_argument("--out", required=True)
    # A different, larger model family than the generator, so it isn't grading itself.
    p.add_argument("--judge-model", default="gemma4:26b")
    p = sub.add_parser("review-sheet"); p.add_argument("--out", required=True)
    p = sub.add_parser("score"); p.add_argument("--out", required=True)
    p.add_argument("--judgments", default=None); p.add_argument("--review", default=None)
    args = ap.parse_args()
    {"retrieve": stage_retrieve, "generate": stage_generate, "ingest": stage_ingest,
     "judge-jobs": stage_judge_jobs, "review-sheet": stage_review_sheet, "score": stage_score}[args.stage](args)


if __name__ == "__main__":
    main()
