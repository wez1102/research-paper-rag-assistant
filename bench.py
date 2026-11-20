"""
Latency benchmark, stage by stage: query embedding + dense search, BM25, fusion,
cross-encoder rerank. Generation is measured separately (answer_eval.py records
time-to-first-token and total time per answer), because it runs on a different
machine/process (Ollama) and dwarfs retrieval anyway.

Conditions are printed with the results: CPU model, core count, torch threads,
index size, and the retrieval settings. Run it on an otherwise idle machine; the
evaluate.py latency columns are measured while other jobs may be running and are
not a benchmark.

    python bench.py                          # app index, 144 questions, both scopes
    python bench.py --split dev --repeat 2   # dev index, every question twice
    RERANK_HOW=maxp python bench.py          # compare an optimisation
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from config import settings
from evaluate import config_fingerprint, load_questions
from retrieve import Retriever

STAGES = ("dense_s", "bm25_s", "fusion_s", "rerank_s")


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def machine() -> dict:
    import torch
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"cpu": cpu, "cores": os.cpu_count(), "torch_threads": torch.get_num_threads(),
            "device": settings.device, "python": platform.python_version(), "torch": torch.__version__}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="app")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions")
    ap.add_argument("--out", default=None, help="write a JSON report here")
    args = ap.parse_args()

    questions = load_questions(args.split)
    if args.limit:
        questions = questions[:args.limit]
    t0 = time.perf_counter()
    r = Retriever(collection_name=settings.collection_name(split=args.split), load_reranker=True)
    load_s = time.perf_counter() - t0
    r.retrieve("warm up", mode="hybrid_rerank")          # first call pays one-off costs

    report = {"conditions": {**machine(), "split": args.split, "n_chunks": len(r.ids),
                             "n_questions": len(questions), "repeat": args.repeat,
                             "config": config_fingerprint(), "startup_load_s": round(load_s, 2)},
              "results": {}}
    for scope in ("paper", "corpus"):
        stages = {s: [] for s in STAGES}
        totals, pairs = [], []
        for _ in range(args.repeat):
            for q in questions:
                timings: dict = {}
                t = time.perf_counter()
                r.retrieve(q["question"], mode="hybrid_rerank", paper_id=q["paper_id"] if scope == "paper" else None,
                           timings=timings)
                totals.append(time.perf_counter() - t)
                for s in STAGES:
                    stages[s].append(timings.get(s, 0.0))
                pairs.append(timings.get("rerank_pairs", 0))
        res = {s: {"p50_ms": round(pct(v, 50) * 1000, 1), "p95_ms": round(pct(v, 95) * 1000, 1)}
               for s, v in stages.items()}
        res["total"] = {"p50_ms": round(pct(totals, 50) * 1000, 1), "p95_ms": round(pct(totals, 95) * 1000, 1)}
        res["rerank_pairs_mean"] = round(statistics.mean(pairs), 1)
        report["results"][scope] = res

    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
