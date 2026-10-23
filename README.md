---
title: Paper QA Assistant
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Cited answers from 50 NLP research papers (QASPER)
---

# Research Paper RAG Assistant

Ask questions about 50 NLP research papers and get answers where every claim is cited as
**[paper title, section]**. Under the hood: section-aware chunking, ChromaDB, hybrid
BM25 + dense retrieval with reciprocal rank fusion, a cross-encoder reranker, and an
evaluation harness that measures retrieval quality on 144 human-labelled questions
from the [QASPER](https://huggingface.co/datasets/allenai/qasper) dataset.

**Run it yourself:** `uvicorn server:app` after `python ingest.py` (see [Run it locally](#run-it-locally)). There's no hosted demo yet: see [Deploy](#deploy-to-hugging-face-spaces).

![The web app: an answer with footnote citations, and the cited passage in the margin with its best supporting sentence highlighted](docs/screenshot.png)

### The web app

`uvicorn server:app` serves a single-page UI (plain HTML/CSS/JS in `static/`, no build step) on top of a small FastAPI API:

- **Pick any local model.** The model menu lists whatever is installed in your Ollama right now (`/api/tags`), plus any API provider you've configured, plus a no-LLM fallback. You can switch per question. If Ollama isn't running, the menu says so and how to start it.
- **Streaming.** The passages appear in the margin as soon as retrieval finishes (about 2 s). Then the answer streams in token by token over Server-Sent Events.
- **Citations you can check.** `[n]` markers become footnote numbers. Hovering one lights up its passage in the margin, and the sentence in that passage that best supports the answer gets highlighted. Uncited passages fold away. "Copy with citations" copies the answer in the `[paper title, section]` format.
- **Honest refusals.** When it declines, it says why: nothing relevant was found (and no model was asked), the model found no answer in the passages, or the answer came without any source.
- **Search controls.** Dense / hybrid / hybrid + rerank, a single-paper filter, the number of passages, and a "How well does it work?" panel that reads the numbers straight from `results/`.
- **Light and dark** themes follow the system, the layout works down to phone width (sources fold under each answer), keyboard focus is visible, and reduced motion is respected. The fonts (Source Serif 4, Schibsted Grotesk) are self-hosted, so the page makes no third-party requests.

The answer path is `generate.answer_events()`. It uses the same retrieval, refusal floor and citation repair that `evaluate.py` scores, so the demo and the benchmark can't drift apart.

<img src="docs/screenshot-mobile.png" alt="The same app on a phone in dark mode" width="300">

## Results

Retrieval quality on **144 questions** over the **50-paper / 1,083-chunk** index. A retrieved
chunk counts as correct when it contains one of the paragraphs the QASPER annotators marked
as evidence. MRR is computed over the top 5.

**Paper scope.** The search is limited to the paper the question was asked about (the "Limit to one paper" filter in the app). This is the setting QASPER was built for.

| Retrieval mode | Hit@1 | Hit@3 | Hit@5 | MRR | Median latency (s) |
|---|---|---|---|---|---|
| Dense (bge-small) | 31.9% | 59.7% | 72.9% | 0.468 | 0.012 |
| Hybrid (BM25 + dense, RRF) | 31.9% | 61.1% | 75.7% | 0.480 | 0.013 |
| **Hybrid + cross-encoder rerank** | **41.0%** | **68.1%** | **84.7%** | **0.564** | 2.146 |

**Corpus scope.** The search covers all 50 papers at once. This is much harder, because many QASPER questions
("What baselines do they compare to?") never say which paper they mean.

| Retrieval mode | Hit@1 | Hit@3 | Hit@5 | MRR | Median latency (s) |
|---|---|---|---|---|---|
| Dense (bge-small) | 14.6% | 26.4% | 34.7% | 0.214 | 0.012 |
| Hybrid (BM25 + dense, RRF) | 16.7% | 29.9% | 36.1% | 0.238 | 0.012 |
| **Hybrid + cross-encoder rerank** | **20.8%** | **36.8%** | **47.9%** | **0.304** | 2.627 |

**Takeaways**

- Hybrid + rerank lifts **Hit@5 from 72.9% → 84.7%** in paper scope and **34.7% → 47.9%** in corpus scope (+38% relative). MRR goes from 0.468 to 0.564.
- BM25 on its own adds a modest gain (+1.4 to +2.8 Hit@5 points). The cross-encoder accounts for most of the improvement.
- The reranker costs about 2–2.5 s per query on a 4-core CPU, and a GPU or Apple's MPS cuts that a lot. `rerank_candidates` in `config.py` trades speed for recall.

### How the reranker was tuned (without overfitting the benchmark)

The first version of the reranker (whole-chunk scoring, top 20) reached 80.6% / 43.8% Hit@5. Tuning more knobs on the same 144 questions would just fit those 144 questions. So `experiments.py` uses a separate **dev set**, QASPER validation papers 50–99 (109 questions, never shown to the app). It picks every setting there and only then reports on the 144 test questions.

Three changes held up on dev and were kept:

1. **Best-passage scoring (MaxP).** A 450-token chunk is a lot for a cross-encoder trained on short MS MARCO passages, and one relevant sentence gets diluted. Each chunk is also split into paragraph-sized passages, and the chunk's score becomes the better of its whole-chunk score and its best-passage score.
2. **Prior fusion.** The reranked order is blended (RRF, weight 0.5) with the first-stage hybrid order. A noisy cross-encoder score then can't bury a chunk that both BM25 and dense search ranked highly.
3. **Rerank depth 30** instead of 20.

| | Dev Hit@5 (paper / corpus) | Test Hit@5 (paper / corpus) | Test MRR (paper / corpus) |
|---|---|---|---|
| Original reranker | 87.2% / 46.8% | 80.6% / 43.8% | 0.536 / 0.285 |
| **Tuned reranker** | **89.9% / 50.5%** | **84.7% / 47.9%** | **0.564 / 0.304** |

The test gains (+4.1 points in both scopes) are close to the dev gains (+2.7 / +3.7), which suggests they're real rather than noise. Full grid: `results/ablation.md` and `results/ablation_all.csv`. One idea helped only without the reranker. Fusing passage-level dense and BM25 rankings into the first stage (`hybrid_maxp_dense_heavy`) beats plain hybrid on its own, but with the cross-encoder on top, plain hybrid candidates did better on dev, so that is what the app uses.

**Chunk-size experiment** (original reranker):

| Chunk size | Hit@5 (paper) | MRR (paper) | Hit@5 (corpus) | MRR (corpus) |
|---|---|---|---|---|
| 256 tokens | 75.0% | 0.496 | 43.1% | 0.285 |
| **450 tokens** (default) | **80.6%** | **0.536** | **43.8%** | 0.285 |

Bigger chunks win in paper scope. QASPER evidence is often spread across neighbouring sentences, and a 256-token window cuts it apart.

### Answer quality

Same 30 sampled questions (seed 42) plus 10 off-topic ones, all with the tuned reranker, qwen3:8b running in Ollama on a MacBook Pro:

| Generator | Token F1 | Citation accuracy | Refused (answerable Qs) | Refused (off-topic Qs) | Median generation time |
|---|---|---|---|---|---|
| Extractive baseline (no LLM) | 0.151 | 100% | 0% | 90% | < 1 ms |
| qwen3:8b, prompt v1 | 0.297 | 100% | 3% (1/30) | 100% | 10.9 s |
| **qwen3:8b, prompt v2 + citation repair** | **0.385** | **100%** | **0%** | **100%** | 10.1 s |

The prompt was chosen on the dev papers too (`results/answers_ablation.md`):

| On dev (papers 50–99) | Token F1 | Refused (answerable) |
|---|---|---|
| v1 | 0.254 | 10% |
| v2 (answer-first) | 0.318 | 10% |
| v2 + citation repair | 0.324 | 7% |

What changed, and why:

- **Answer-first prompt (v2).** The v1 answers were correct but wordy. QASPER references are short ("100", "Yes", "Unif and Stopword"), and so is what a reader wants first. v2 asks for the shortest direct answer, using the sources' own terms, followed by an optional sentence of support. It also says outright that a source answering *in other words* counts. (The old "how many participants?" miss, where the source said "100 crowdworkers", is gone with both prompts once the tuned retriever is in place; v2's answer is simply "100 [2]".)
- **Citation repair.** Sometimes the model gives the right answer but copies the paper's own reference marker ("based on RoBERTa BIBREF2.") instead of a source number. The old code treated that as an unsupported answer and refused. Now the model gets one follow-up turn asking it to cite a numbered source or say it can't. On dev that rescued one answer. On test it never fired. It costs an extra LLM call only in that case, and the code never adds a citation itself.
- **Refusals still work.** The reranker floor (`MIN_RERANK_SCORE`, re-calibrated to −9 for the new scoring) stops 9 of the 10 off-topic questions before any LLM call. The LLM refuses the 10th on its own, which is why the extractive baseline, with no LLM, only gets 9/10.
- **Token F1 is harsh.** It compares words with a short reference span, so a correct answer plus one sentence of context still scores well below 1. Treat it as a relative measure between versions.

Files: `results/answers_ollama_v{1,2}.csv` (per question) and `results/answers_summary_*.csv`. `python evaluate.py --skip-retrieval --answers 30 --provider ollama` regenerates them; `PROMPT_VERSION=v1` or `CITE_REPAIR=0` turns the changes off. The first-round run (old retriever, v1 prompt, F1 0.300) is kept in `results/history/`.

## How it works

```mermaid
flowchart LR
    subgraph Offline["Offline (ingest.py, run once)"]
        A[QASPER validation split<br/>first 50 papers] --> B[Flatten to paragraphs<br/>+ section names]
        B --> C[Section-aware chunking<br/>450 tokens, 50 overlap]
        C --> D[bge-small-en-v1.5<br/>embeddings]
        D --> E[(ChromaDB<br/>persistent)]
        B --> Q[questions.jsonl<br/>gold evidence paragraphs]
    end

    subgraph Query["Query path (retrieve.py + generate.py)"]
        U[Question] --> DS[Dense search<br/>top 50]
        U --> BM[BM25 search<br/>top 50]
        E --> DS
        E --> BM
        DS --> F[Reciprocal rank fusion]
        BM --> F
        F --> R[Cross-encoder rerank<br/>best-passage scores<br/>top 30 → top 5]
        R --> G{Relevant enough?}
        G -->|no| REF["I can't find this in the papers."]
        G -->|yes| L[LLM with numbered sources]
        L --> CI["Answer with [title, section] citations"]
    end

    Q --> EV["evaluate.py<br/>Hit@k, MRR, latency"]
    R --> EV
```

The app and the evaluation script call the **same** `Retriever.retrieve()` function, so the reported scores are the scores of the system people actually use.

- **Chunking (`ingest.py`).** Text is split per section and never across sections or papers. Chunk size is measured with the embedding model's own tokenizer, so chunks really fit in its 512-token window. Character offsets preserve the original casing. Each chunk records which paragraphs it holds, which is how evaluation stays automatic.
- **Retrieval (`retrieve.py`).** Dense search uses Chroma with cosine similarity on bge-small, with the query instruction prefix the model was trained with. BM25 uses `rank_bm25` over the same "title. section. text" string. The two ranked lists are merged with reciprocal rank fusion (k = 60). RRF uses only ranks, so the two very different score scales never need to be normalised against each other. Reranking uses `cross-encoder/ms-marco-MiniLM-L-6-v2` on the top 30. Each chunk scores the better of its whole text and its best paragraph, and that order is blended back with the hybrid order (see the tuning section above).
- **Generation (`generate.py`).** The LLM sees the chunks as numbered sources and must cite `[n]` after each claim. Small models get numbers right far more often than long titles. The code then rewrites `[n]` as `[paper title, section]` and drops any citation that doesn't match a retrieved chunk. The system says *"I can't find this in the papers"* in three cases: the reranker's best score is below a calibrated floor, the model says so, or the model still has no valid citation after one repair turn.
- **LLM backends.** Ollama (free, local), any OpenAI-compatible API (OpenAI, Groq's free tier, …), Anthropic, or an **extractive** fallback that quotes the best sentences and needs no key at all. `LLM_PROVIDER=auto` picks whatever is configured.

## Run it locally

Requires Python 3.10+ and about 2 GB of disk (PyTorch plus ~230 MB of models). CPU only.

```bash
git clone <your-repo-url> && cd RAG_Project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # then pick an LLM (see below), or leave it on auto

python ingest.py              # download QASPER + models, build the index (~2 min on a laptop)
python evaluate.py            # retrieval eval -> results/results.csv, results.md
uvicorn server:app            # open http://localhost:8000
# (the simpler Streamlit page still works: streamlit run app.py)
```

Other useful commands:

```bash
python generate.py "Which datasets are used for Armenian NER?"   # ask from the terminal
python retrieve.py "BLEU on Europarl" --mode hybrid               # inspect raw retrieval
python ingest.py --chunk-size 256 --chunk-overlap 32               # build a second index...
python evaluate.py --chunk-size 256                                # ...and compare it
python evaluate.py --modes hybrid_rerank --scopes paper            # re-run one row only
python evaluate.py --skip-retrieval --answers 30                   # answer quality with your LLM
python experiments.py                                              # dev/test retrieval ablation (cached, resumable)
python experiments.py --budget 150                                 # ...or in 150-second slices on a slow machine
pytest -q                                                          # 23 tests: logic, API (fake LLM), headless app
```

The ablation works with other models too, e.g. on a machine with a GPU or Apple silicon:
`EMBED_MODEL=BAAI/bge-base-en-v1.5 RERANK_MODEL=BAAI/bge-reranker-base python experiments.py`
(rebuild the index with the same `EMBED_MODEL` before the app uses it).

**Picking an LLM**

| Option | Cost | Setup |
|---|---|---|
| Ollama (local) | free | install [Ollama](https://ollama.com) and `ollama pull qwen3:8b` (or any chat model). The web app finds installed models on its own; for the CLI and `evaluate.py`, set `LLM_PROVIDER=ollama` |
| Groq (hosted, OpenAI-compatible) | free tier | `OPENAI_API_KEY=<groq key>`, `OPENAI_BASE_URL=https://api.groq.com/openai/v1`, `LLM_MODEL=` any model from Groq's current list |
| OpenAI / Anthropic | a few cents for the whole eval | set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` |
| Extractive | free | nothing; this is the fallback when no LLM is configured |

## Deploy to Hugging Face Spaces

Hugging Face now requires a PRO plan ($9/month) to create Docker or Gradio Spaces; only static Spaces are free. Ollama can't run on a Space either, so a hosted demo uses an API for answers (Groq's free tier works well, e.g. `LLM_MODEL=llama-3.3-70b-versatile`) or falls back to extractive answers.

1. Create a new Space → **Docker** → *Blank*.
2. Push this repo to the Space (`git remote add space https://huggingface.co/spaces/<you>/<space>` then `git push space main`). The YAML block at the top of this README tells Spaces to use the Dockerfile on port 7860.
3. In **Settings → Variables and secrets**, add the secret `OPENAI_API_KEY` (your Groq key) and the variables `OPENAI_BASE_URL=https://api.groq.com/openai/v1` and `LLM_MODEL=llama-3.3-70b-versatile` (or set `ANTHROPIC_API_KEY`).
4. The Docker build runs `python ingest.py`, so the index is baked into the image and the Space starts ready to answer.
5. The Space serves the same web app. Its model menu shows your API model and the no-LLM option, and the Ollama entry explains that it isn't available there. For a paid API the server always uses the model you configured, whatever the browser asks for, so a public demo can't be pointed at an expensive model.

## Project structure

```
config.py          all settings (models, chunk size, paths, LLM), overridable via env/.env
ingest.py          QASPER -> paragraphs -> chunks -> embeddings -> ChromaDB (+ questions.jsonl)
retrieve.py        dense / BM25 / RRF hybrid / cross-encoder rerank behind one retrieve()
generate.py        prompt, LLM backends, citation rewriting, refusal logic, CLI
evaluate.py        Hit@1/3/5, MRR, latency per mode & scope; answer F1 + citation accuracy
experiments.py     dev-tuned / test-reported retrieval ablation (MaxP, fusion, rerank depth, prior)
server.py          FastAPI app: model discovery, streaming /api/ask (SSE), benchmark numbers
static/            the web UI: index.html, app.css, app.js (no framework, no build step), fonts
app.py             the original Streamlit UI (still works, simpler)
tests/             unit tests, API tests with a fake LLM (test_server.py), headless Streamlit test
docs/              screenshots
data/questions.jsonl   the 144 evaluation questions with gold evidence paragraph ids
results/           results.csv/.md, per-question breakdowns, ablation*.{md,csv}, answer checks
Dockerfile         CPU-only image for Hugging Face Spaces
```

## Notes and decisions

- **Why QASPER?** It gives full paper text *and* questions with annotated evidence paragraphs, so retrieval can be scored automatically with no hand labelling. 156 questions belong to the first 50 validation papers. 5 of them are unanswerable, 6 have only figures or tables as evidence, and 1 has evidence that doesn't map to a paragraph, which leaves 144.
- **Loading QASPER.** `allenai/qasper` is an old script-based dataset, and recent `datasets` versions refuse to run it. `ingest.py` reads the Hub's auto-converted Parquet file (`refs/convert/parquet`) instead.
- **Scoring is strict.** Thanks to the 50-token overlap, a chunk may hold the tail of the previous paragraph. A chunk only "claims" a paragraph when it holds at least half of it (or at least 100 of its tokens), so overlap can't inflate hit rates.
- **Reproducibility.** Dependencies are pinned, the corpus is the first 50 papers (no random sampling), the answer-eval sample uses seed 42, and LLM calls use temperature 0 where the API supports it. `python ingest.py` always rebuilds the collection from scratch.
- **Latency** was measured on a 4-core ARM Linux VM with models already loaded. Expect similar or better on a recent laptop.

## What I'd try next

- **Query rewriting.** Rewrite vague questions ("what baselines do they use?") with the LLM, or route them to the most likely paper first. That attacks the corpus-scope gap directly.
- **Answer faithfulness checks.** Verify each cited sentence against its chunk with an NLI model.
- **A bigger corpus and a better embedder** (bge-base, e5), plus parent-document retrieval for long sections.
- **A larger local LLM** (e.g. qwen3:14b) on a machine with more memory, run through the same dev/test split as the prompt comparison.

## Licence and credits

Code: MIT. QASPER is © AllenAI, released under CC BY 4.0 (Dasigi et al., 2021, [arXiv:2105.03011](https://arxiv.org/abs/2105.03011)).
Models: [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2).
