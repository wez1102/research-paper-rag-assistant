# Round-2 results (Nov 2026)

Round-1 files in `results/` (one level up) are unchanged. Everything here was produced in
round 2. See `docs/EVALUATION.md` for what the numbers mean.

| Folder / file | What | Split | Notes |
|---|---|---|---|
| `repro_app/` | round-1 code re-run unchanged | app | reproduces round 1 except one question in corpus Hit@3 |
| `dev/baseline/` | round-1 config, HNSW dense search | dev | + `candidate_recall.csv`, `corpus_failure_analysis.csv` |
| `dev/exact/` | + exact dense search | dev | reranker scores cached: latency columns not meaningful |
| `dev/depth50/`, `dev/maxp/`, `dev/maxp_len256/` | one change each | dev | cached scores: latency columns not meaningful |
| `final/` | frozen config, all modes, both scopes | **final (held out)** | + candidate recall and failure analysis |
| `final_max_scoring/` | same, with round-1 `max` scoring | final | the speed/quality trade-off |
| `app/` | round-2 code on the app papers | app | comparison with round 1 |
| `bench/dev_*.json` | `bench.py` per-stage latency p50/p95 | dev, 60 questions | idle 2-core x86; conditions inside each file |
| `clarify_detector.json` | "which paper?" rule vs specificity labels | dev, final | |
| `answers/dev_v2/`, `answers/dev_v3/` | answer runs, prompt v2 vs v3 | dev answer set | reviews in `eval/review_dev_*.json` |
| `answers/final/` | answer run, shipped config | **final answer set** | `answer_eval_summary.json`, `answer_eval_per_question.csv`, `review_sheet.md`, judge outputs (qwen3:14b), relay outputs with SHA-1s |
