"""
Step 4: measure how good retrieval actually is.

For every QASPER question we know which paragraphs the human annotators used as
evidence. A retrieved chunk is a "hit" if it contains one of those paragraphs.
From that we compute, per retrieval mode:

  Hit@1 / Hit@3 / Hit@5  - share of questions with a correct chunk in the top k
  MRR                    - mean of 1/rank of the first correct chunk (0 if none in top 5)
  median latency         - seconds per retrieval call (models already loaded)

Two search scopes are reported:
  corpus  - search all 50 papers at once. Hard mode: QASPER questions like
            "What baselines do they compare to?" don't say which paper they mean.
  paper   - search only inside the paper the question was asked about (the same
            thing as picking a paper in the app's filter). This is the setting
            QASPER was designed for.

Optionally (--answers N) it also scores generated answers on a sample of N
questions: token F1 against the reference answers (QASPER's standard metric),
citation accuracy, and how often the system refused.

Usage:
    python evaluate.py                    # retrieval eval, writes results/results.csv
    python evaluate.py --chunk-size 256   # same, against the 256-token index
    python evaluate.py --answers 30       # also run the answer-quality check (needs an LLM)

    # Round-2 protocol (see docs/EVALUATION.md): tune on dev, report once on final.
    python ingest.py --split dev && python evaluate.py --split dev --out results/v2/dev
    python ingest.py --split final && python evaluate.py --split final --out results/v2/final
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import string
import time
from collections import Counter
from pathlib import Path

import numpy as np

import pandas as pd

from config import settings
from retrieve import MODES, Retriever, RetrievedChunk

KS = (1, 3, 5)

# Questions the papers obviously can't answer. A grounded system should refuse all of them.
OFF_TOPIC = [
    "What is the capital of France?",
    "How do I bake sourdough bread?",
    "Who won the 2018 FIFA World Cup?",
    "What is the boiling point of water at sea level?",
    "Which stocks should I buy this year?",
    "How tall is Mount Everest?",
    "What is the recipe for chocolate cake?",
    "Who painted the Mona Lisa?",
    "What time does the sun set in Toronto in winter?",
    "How many legs does a spider have?",
]
SCOPES = ("corpus", "paper")
MODE_LABELS = {"dense": "Dense", "hybrid": "Hybrid", "hybrid_rerank": "Hybrid + rerank"}


# Columns that look numeric but are ids. Without this, pandas reads the paper id
# "1710.07960" back as the float 1710.0796 (the round-1 per_question.csv has this bug).
ID_DTYPES = {"question_id": str, "paper_id": str, "config": str}


def load_questions(split: str = "app") -> list[dict]:
    path = settings.questions_path(split)
    if not path.exists():
        raise SystemExit(f"{path.name} is missing - run `python ingest.py --split {split}` first.")
    return [json.loads(line) for line in path.open()]


def first_hit_rank(results: list[RetrievedChunk], q: dict) -> int | None:
    """1-based rank of the first chunk that contains a gold evidence paragraph, or None."""
    gold = set(q["gold_paragraph_ids"])
    for rank, r in enumerate(results, start=1):
        if r.paper_id == q["paper_id"] and gold & set(r.paragraph_ids):
            return rank
    return None


def right_paper_rank(results: list[RetrievedChunk], q: dict) -> int | None:
    """1-based rank of the first chunk from the question's own paper (any passage)."""
    for rank, r in enumerate(results, start=1):
        if r.paper_id == q["paper_id"]:
            return rank
    return None


def summarize(ranks: list[int | None], latencies: list[float],
              paper_ranks: list[int | None] | None = None) -> dict:
    n = len(ranks)
    row = {f"hit@{k}": sum(1 for r in ranks if r is not None and r <= k) / n for k in KS}
    row["mrr"] = sum(1.0 / r for r in ranks if r is not None) / n
    row["median_latency_s"] = statistics.median(latencies)
    row["p95_latency_s"] = float(np.percentile(latencies, 95))
    if paper_ranks is not None:
        # "Did we at least find the right paper?" Separates paper identification
        # from passage ranking when searching the whole corpus.
        row["paper@1"] = sum(1 for r in paper_ranks if r == 1) / n
        row["paper@5"] = sum(1 for r in paper_ranks if r is not None) / n
    row["n_questions"] = n
    return row


def config_fingerprint() -> str:
    """Everything that changes retrieval results. Rows from other configs are never mixed in."""
    from ingest import model_tag
    return (f"{model_tag(settings.embed_model)}|{model_tag(settings.rerank_model)}|cand={settings.rerank_candidates}|"
            f"how={settings.rerank_how}|prior={settings.rerank_prior}|rrfw={','.join(map(str, settings.rrf_weights))}|"
            f"pboost={settings.paper_boost}|maxlen={settings.rerank_max_length}")


def run_retrieval_eval(retriever: Retriever, questions: list[dict], k: int = 5,
                       modes=MODES, scopes=SCOPES) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, per_question = [], []
    for mode in modes:
        # One throwaway query so model warm-up time doesn't pollute the latency numbers.
        retriever.retrieve("warm up", mode=mode, k=k)
        for scope in scopes:
            ranks, latencies, paper_ranks = [], [], []
            for q in questions:
                timings: dict = {}
                t0 = time.perf_counter()
                results = retriever.retrieve(q["question"], mode=mode, k=k,
                                             paper_id=q["paper_id"] if scope == "paper" else None,
                                             timings=timings)
                latencies.append(time.perf_counter() - t0)
                rank = first_hit_rank(results, q)
                ranks.append(rank)
                paper_ranks.append(right_paper_rank(results, q))
                per_question.append({"mode": mode, "scope": scope, "config": config_fingerprint(),
                                     "question_id": q["question_id"],
                                     "question": q["question"], "paper_id": q["paper_id"],
                                     "first_hit_rank": rank, "right_paper_rank": paper_ranks[-1],
                                     "latency_s": latencies[-1],
                                     **{k_: round(v, 5) for k_, v in timings.items()},
                                     "retrieved": " | ".join(r.chunk_id for r in results)})
            row = {"mode": mode, "scope": scope,
                   **summarize(ranks, latencies, paper_ranks if scope == "corpus" else None)}
            summary.append(row)
            extra = f"  paper@5={row['paper@5']:.3f}" if "paper@5" in row else ""
            print(f"{mode:>14} / {scope:<6}  hit@1={row['hit@1']:.3f}  hit@3={row['hit@3']:.3f}  "
                  f"hit@5={row['hit@5']:.3f}  mrr={row['mrr']:.3f}{extra}  "
                  f"p50={row['median_latency_s']:.3f}s  p95={row['p95_latency_s']:.3f}s")
    return pd.DataFrame(summary), pd.DataFrame(per_question)


def candidate_recall(retriever: Retriever, questions: list[dict], depths=(5, 10, 20, 30, 50, 100)) -> pd.DataFrame:
    """
    How deep does the gold passage sit in the first-stage (hybrid) list? If it isn't
    in the top N at all, no reranker looking at N candidates can recover it. This is
    what tells us whether rerank depth is worth paying for.
    """
    rows = []
    for scope in SCOPES:
        found = {d: 0 for d in depths}
        for q in questions:
            hits = retriever.hybrid_search(q["question"], max(depths),
                                           paper_id=q["paper_id"] if scope == "paper" else None)
            res = [retriever._to_result(cid, s, "rrf") for cid, s in hits]
            r = first_hit_rank(res, q)
            for d in depths:
                found[d] += r is not None and r <= d
        rows.append({"scope": scope, **{f"recall@{d}": found[d] / len(questions) for d in depths}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Answer quality
# ---------------------------------------------------------------------------

def _normalize_answer(s: str) -> list[str]:
    """SQuAD-style normalisation: lowercase, drop punctuation and articles."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return s.split()


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = _normalize_answer(prediction), _normalize_answer(reference)
    if not pred or not ref:
        return float(pred == ref)
    common = Counter(pred) & Counter(ref)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pred), same / len(ref)
    return 2 * precision * recall / (precision + recall)


def run_answer_eval(retriever: Retriever, questions: list[dict], n: int, mode: str, scope: str,
                    provider: str | None) -> tuple[pd.DataFrame, dict]:
    from generate import answer_question

    rng = random.Random(settings.seed)          # fixed seed -> same 30 questions every run
    sample = rng.sample(questions, min(n, len(questions)))
    rows = []
    for i, q in enumerate(sample, start=1):
        ans = answer_question(q["question"], retriever, mode=mode,
                              paper_id=q["paper_id"] if scope == "paper" else None, provider=provider)
        # Score the model's own words, not the long citation strings we add afterwards.
        plain = re.sub(r"\[\d+(?:\s*[,;]\s*\d+)*\]", "", ans.raw)
        f1 = 0.0 if ans.refused else max((token_f1(plain, ref) for ref in q["references"]), default=0.0)
        rows.append({"question_id": q["question_id"], "question": q["question"], "answer": ans.text,
                     "references": " || ".join(q["references"]), "token_f1": f1, "refused": ans.refused,
                     "citation_accuracy": ans.citation_accuracy, "provider": ans.provider, "model": ans.model,
                     "generation_s": ans.timings.get("generation_s", 0.0)})
        print(f"  [{i}/{len(sample)}] f1={f1:.2f} refused={ans.refused}  {q['question'][:70]}")
    df = pd.DataFrame(rows)
    cited = df["citation_accuracy"].dropna()

    # FR-8 check: does it say "I can't find this" when it should?
    off_refused = [answer_question(q, retriever, mode=mode, provider=provider).refused for q in OFF_TOPIC]
    summary = {
        "mode": mode, "scope": scope, "provider": df["provider"].iloc[0], "model": df["model"].iloc[0],
        "n_questions": len(df),
        "token_f1": df["token_f1"].mean(),
        "citation_accuracy": cited.mean() if len(cited) else None,
        "refusal_rate": df["refused"].mean(),
        "off_topic_refusal_rate": sum(off_refused) / len(off_refused),
        "median_generation_s": df["generation_s"].median(),
    }
    return df, summary


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def upsert_csv(path, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """
    Merge `new` rows into an existing CSV, replacing rows with the same keys.
    This lets you run one mode at a time (e.g. `--modes hybrid_rerank` on a slow
    laptop) without losing the numbers from earlier runs.
    """
    if path.exists():
        old = pd.read_csv(path, dtype=ID_DTYPES)
        merged_keys = old[keys].astype(str).agg("|".join, axis=1)
        new_keys = set(new[keys].astype(str).agg("|".join, axis=1))
        old = old[~merged_keys.isin(new_keys)]
        new = pd.concat([old, new], ignore_index=True)
    # Keep a stable, readable row order: dense -> hybrid -> rerank, corpus before paper.
    order = {m: i for i, m in enumerate(MODES)}
    new = new.sort_values(by=["scope", "mode"], key=lambda col: col.map(order) if col.name == "mode" else col)
    new.to_csv(path, index=False)
    return new


def markdown_table(summary: pd.DataFrame, scope: str) -> str:
    """The README table, generated straight from the numbers so it can't drift."""
    has_p95 = "p95_latency_s" in summary.columns
    has_paper = scope == "corpus" and "paper@5" in summary.columns
    head = "| Retrieval mode | Hit@1 | Hit@3 | Hit@5 | MRR |" + (" Right paper in top 5 |" if has_paper else "")
    head += " Median latency (s) |" + (" p95 latency (s) |" if has_p95 else "")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for _, r in summary[summary["scope"] == scope].iterrows():
        line = (f"| {MODE_LABELS[r['mode']]} | {r['hit@1']:.1%} | {r['hit@3']:.1%} | {r['hit@5']:.1%} "
                f"| {r['mrr']:.3f} |")
        if has_paper:
            line += f" {r['paper@5']:.1%} |"
        line += f" {r['median_latency_s']:.3f} |"
        if has_p95:
            line += f" {r['p95_latency_s']:.3f} |"
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval (and optionally answers) on QASPER.")
    parser.add_argument("--split", choices=list(settings.splits), default="app",
                        help="Which paper slice to evaluate on (build it first with ingest.py --split).")
    parser.add_argument("--out", default=None, help="Where to write results (default: results/).")
    parser.add_argument("--ce-cache", default=None,
                        help="JSON file to memoise cross-encoder scores in (speeds up depth/weight sweeps).")
    parser.add_argument("--candidate-recall", action="store_true",
                        help="Also report how deep the gold passage sits in the first-stage list.")
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size,
                        help="Which index to evaluate (must have been built with ingest.py --chunk-size).")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--scopes", nargs="+", choices=SCOPES, default=list(SCOPES))
    parser.add_argument("--skip-retrieval", action="store_true", help="Only run --answers.")
    parser.add_argument("--part", default="", help="e.g. 1/3 = run only the first third of the questions. "
                        "Parts merge, and the summary appears once every part has run (handy on slow machines).")
    parser.add_argument("--answers", type=int, default=0, help="Also score N generated answers.")
    parser.add_argument("--answer-mode", choices=MODES, default="hybrid_rerank")
    parser.add_argument("--answer-scope", choices=SCOPES, default="paper")
    parser.add_argument("--provider", default=None, help="Force an LLM provider for --answers.")
    args = parser.parse_args()

    random.seed(settings.seed)
    out_dir = Path(args.out) if args.out else settings.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = load_questions(args.split)
    collection = settings.collection_name(args.chunk_size, args.split)
    print(f"Evaluating {len(questions)} questions ({args.split} split) against index '{collection}'")
    print(f"Config: {config_fingerprint()}")

    cache: dict | None = None
    if args.ce_cache:
        cache_path = Path(args.ce_cache)
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    need_reranker = "hybrid_rerank" in args.modes or args.answer_mode == "hybrid_rerank"
    retriever = Retriever(collection_name=collection, load_reranker=need_reranker, score_cache=cache)

    if args.candidate_recall:
        cr = candidate_recall(retriever, questions)
        cr.to_csv(out_dir / "candidate_recall.csv", index=False)
        print(cr.to_string(index=False))

    if not args.skip_retrieval:
        run_qs = questions
        if args.part:
            i, n = (int(x) for x in args.part.split("/"))
            size = -(-len(questions) // n)                     # ceiling division
            run_qs = questions[(i - 1) * size:i * size]
            print(f"Part {i}/{n}: questions {(i - 1) * size + 1}-{(i - 1) * size + len(run_qs)}")
        _, per_q = run_retrieval_eval(retriever, run_qs, modes=args.modes, scopes=args.scopes)

        if cache is not None:
            Path(args.ce_cache).write_text(json.dumps(cache))
        suffix = "" if args.chunk_size == settings.chunk_size else f"_chunk{args.chunk_size}"
        per_all = upsert_csv(out_dir / f"per_question{suffix}.csv", per_q,
                             ["mode", "scope", "question_id"])

        # Rebuild the summary from per-question rows, only where all questions were
        # run with the current config - so --part runs add up to the same numbers.
        wanted, rows = {q["question_id"] for q in questions}, []
        for (mode, scope), g in per_all.groupby(["mode", "scope"]):
            g = g[(g["config"] == config_fingerprint()) & g["question_id"].astype(str).isin(wanted)]
            if mode not in args.modes or scope not in args.scopes:
                continue
            if len(g) < len(questions):
                print(f"{mode}/{scope}: {len(g)}/{len(questions)} questions done with this config")
                continue
            ranks = [None if pd.isna(r) else int(r) for r in g["first_hit_rank"]]
            p_ranks = None
            if scope == "corpus" and "right_paper_rank" in g:
                p_ranks = [None if pd.isna(r) else int(r) for r in g["right_paper_rank"]]
            rows.append({"split": args.split, "chunk_size": args.chunk_size, "mode": mode, "scope": scope,
                         **summarize(ranks, list(g["latency_s"]), p_ranks), "config": config_fingerprint()})
        results_path = out_dir / f"results{suffix}.csv"
        if rows:
            summary = upsert_csv(results_path, pd.DataFrame(rows), ["mode", "scope"])
        elif results_path.exists():
            summary = pd.read_csv(results_path)
        else:
            return
        with (out_dir / f"results{suffix}.md").open("w") as f:
            for scope in SCOPES:
                if scope in set(summary["scope"]):
                    f.write(f"### Scope: {scope} ({len(questions)} questions, {args.split} split, "
                            f"chunk size {args.chunk_size})\n\n")
                    f.write(markdown_table(summary, scope) + "\n\n")
        print(f"\nUpdated results{suffix}.csv / per_question{suffix}.csv / results{suffix}.md in {out_dir}")

    if args.answers:
        print(f"\nAnswer evaluation on {args.answers} questions ({args.answer_mode}, scope={args.answer_scope})")
        df, ans_summary = run_answer_eval(retriever, questions, args.answers, args.answer_mode,
                                          args.answer_scope, args.provider)
        # One file per provider, so the no-LLM baseline and an LLM run can sit side by side.
        tag = ans_summary["provider"]
        if tag != "extractive":
            tag += f"_{settings.prompt_version}"      # e.g. answers_ollama_v2.csv
            ans_summary["prompt_version"] = settings.prompt_version
        df.to_csv(out_dir / f"answers_{tag}.csv", index=False)
        pd.DataFrame([ans_summary]).to_csv(out_dir / f"answers_summary_{tag}.csv", index=False)
        print(json.dumps(ans_summary, indent=2, default=str))


if __name__ == "__main__":
    main()
