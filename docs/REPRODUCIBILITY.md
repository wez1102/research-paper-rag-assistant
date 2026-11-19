# Reproducing the results

Everything below was run in the round-2 session (Nov 2026). The round-1 numbers (Oct 2026)
are kept unchanged in `results/` and `results/history/`; round-2 numbers live in
`results/v2/`. Nothing in `results/v2/` overwrites a round-1 file.

## Environment

| Item | Round-2 value |
|---|---|
| Python | 3.11.18 (CI: 3.11) |
| torch | 2.9.1 (CPU; the session used the PyPI wheel, the Dockerfile uses the CPU-only wheel) |
| sentence-transformers / transformers | 6.2.0 / 5.20.2 |
| chromadb | 1.5.9 |
| numpy / pandas | 2.2.6 / 2.3.3 |
| everything else | pinned in `requirements.txt` / `requirements-dev.txt` |
| Retrieval hardware | 2-core Intel Xeon @ 2.10 GHz, 7 GB RAM, Linux x86_64, CPU only, 2 torch threads |
| Generation hardware | the owner's MacBook Pro (Apple silicon), Ollama, `qwen3:8b` Q4_K_M, digest `500a1f067a9f`; judge `gemma4:26b` Q4_K_M |
| Round-1 hardware (for comparison) | 4-core ARM Linux VM (retrieval), same Mac (generation) |

## Data and models (exact files)

| File | SHA-256 |
|---|---|
| QASPER `validation/0000.parquet` (HF `allenai/qasper`, branch `refs/convert/parquet`) | `089781b91c337d348dd9e8b57cc8adc100ed2d9cab84a6127402bcccf1559222` |
| `BAAI/bge-small-en-v1.5` `model.safetensors` | `3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad` |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` `model.safetensors` | `821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae` |

`python scripts/download_assets.py assets/` fetches all three. Check them with
`sha256sum`. Each index stores a manifest hash in its Chroma metadata (shown by `/readyz`):

| Index | Papers (validation rows) | Chunks | Answerable Qs | Unanswerable Qs | Manifest |
|---|---|---|---|---|---|
| `qasper_chunks_450` (app) | 0-49 | 1,083 | 144 | 5 | `133c4f76684d...` |
| `qasper_dev_chunks_450` | 50-99 | 926 | 109 | 4 | `29d3c7c9ac73...` |
| `qasper_final_chunks_450` | 100-149 | 1,008 | 209 | 10 | `dfd2f588d950...` |

The three slices share no paper (`tests/test_ingest.py::test_splits_never_share_papers`).

## Commands

```bash
# 0. environment
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
python scripts/download_assets.py assets/            # or let HF download on first use
export QASPER_PARQUET=assets/qasper/validation.parquet \
       EMBED_MODEL=assets/models/bge-small-en-v1.5 \
       RERANK_MODEL=assets/models/ms-marco-MiniLM-L-6-v2 HF_HUB_OFFLINE=1

# 1. indexes (idempotent: a second run prints "already up to date")
python ingest.py                  # app split, what the web app serves
python ingest.py --split dev
python ingest.py --split final

# 2. offline tests (no index, models or LLM needed for most of them)
pytest -q

# 3. retrieval: tune on dev ...
python evaluate.py --split dev --out results/v2/dev/exact --candidate-recall
python failure_analysis.py --split dev --results results/v2/dev/exact
RERANK_CANDIDATES=50 python evaluate.py --split dev --modes hybrid_rerank --out results/v2/dev/depth50
RERANK_HOW=maxp python evaluate.py --split dev --modes hybrid_rerank --out results/v2/dev/maxp
# ... then report once on final
python evaluate.py --split final --out results/v2/final
python failure_analysis.py --split final --results results/v2/final

# 4. latency, on an otherwise idle machine
python bench.py --split dev --limit 60 --out results/v2/bench/dev_max.json
RERANK_HOW=maxp python bench.py --split dev --limit 60 --out results/v2/bench/dev_maxp.json

# 5. answers (needs Ollama with qwen3:8b; see docs/EVALUATION.md)
python answer_eval.py retrieve --set eval/answer_set_final.jsonl --out results/v2/answers/final
python answer_eval.py generate --out results/v2/answers/final --provider ollama --model qwen3:8b
python answer_eval.py judge-jobs --out results/v2/answers/final   # then run them, see EVALUATION.md
python answer_eval.py score --out results/v2/answers/final --judgments <judge outputs> --review eval/review_final.json
```

`--ce-cache file.json` on `evaluate.py` memoises cross-encoder scores, which makes
sweeps fast. **Latency columns from a cached run are meaningless**; use `bench.py`.

## How the round-2 answers were actually generated

In the round-2 session the Python code ran in a sandbox that couldn't reach Ollama on the
Mac. So `answer_eval.py retrieve` wrote the exact prompts, a browser tab on the Mac
(`eval/relay_runner.js`, at `http://localhost:11434`) sent them to Ollama with the same
options as `generate.py` (`think: false`, temperature 0, seed 42, 400 tokens max), and
`answer_eval.py ingest` read the outputs back. Every prompt's SHA-1 was computed on both
sides, and ingest refuses any mismatch. All prompts matched. On a machine that can reach
Ollama directly, `answer_eval.py generate` does the same in one step.

## Known sources of run-to-run difference

- **Fixed in round 2:** dense search used Chroma's approximate HNSW index, which returned
  different top-50 lists from one process to the next for 59 of 109 dev questions. Dense
  search is now exact (NumPy), and two processes give identical rankings for every dev
  question (`python scripts/check_determinism.py --split dev` exits 0).
- Cross-encoder scores differ in the last float digits between x86 and ARM. The round-1
  app-split numbers reproduced exactly except one question's rank in corpus scope
  (Hit@3 37.5% vs 36.8%). The CI regression test allows a 1-question tolerance for this.
- LLM output: temperature 0 and a fixed seed make Ollama runs repeatable on the same
  machine and model build, but not across Ollama versions or hardware.
