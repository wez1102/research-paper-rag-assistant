# Evaluation (round 2, Nov 2026)

This document covers what was measured in round 2, how, and what the numbers do and don't
show. Round-1 results (Oct 2026) are kept unchanged in the README and `results/`; nothing
here replaces them.

## 1. Protocol: splits by paper

| Split | QASPER validation rows | Papers | Answerable Qs | Unanswerable Qs | Used for |
|---|---|---|---|---|---|
| `app` | 0-49 | 50 | 144 | 5 | the papers the app serves; round-1 "test" |
| `dev` | 50-99 | 50 | 109 | 4 | every round-2 decision |
| `final` | 100-149 | 50 | 209 | 10 | opened once, after the configuration was frozen |

- No paper appears in two splits (`tests/test_ingest.py::test_splits_never_share_papers`).
- **Why a new final split.** In round 1 the `app` papers were the test set, but some choices
  were made by looking at them: the 450-vs-256 chunk-size comparison in the README was run on
  test, and the 30-question answer sample was regenerated after each prompt change. So `app`
  is no longer a clean held-out set. Round 2 reports `app` numbers only for comparison with
  round 1.
- **Frozen before `final` was opened:** exact dense search; hybrid (RRF k=60, equal weights);
  rerank depth 30, `maxp` scoring, prior 0.5; refusal floor −9; "which paper?" rule (6%
  threshold); prompt v2. The `max`-scoring variant was also run on `final`, to report the
  speed/quality trade-off, after `maxp` had already been chosen on dev.

## 2. Retrieval

Hit@k counts a question when one of the top-k chunks contains a gold evidence paragraph from
the question's own paper; MRR is over the top 5. **Right paper@5** (corpus scope only) counts
any chunk from the right paper in the top 5, which separates finding the paper from finding
the passage.

### Final split (held out, 209 questions)

| Mode | Paper Hit@1 | Hit@3 | Hit@5 | MRR | Corpus Hit@1 | Hit@3 | Hit@5 | MRR | Right paper@5 |
|---|---|---|---|---|---|---|---|---|---|
| Dense | 38.3% | 67.9% | 76.1% | 0.528 | 17.2% | 28.7% | 31.6% | 0.227 | 49.8% |
| Hybrid (BM25 + dense, RRF) | 39.7% | 67.0% | 79.4% | 0.543 | 14.4% | 31.6% | 38.8% | 0.229 | 56.9% |
| **Hybrid + rerank (`maxp`, shipped)** | 44.0% | 70.8% | **85.2%** | **0.593** | 22.5% | 36.8% | **45.0%** | **0.305** | 62.2% |
| Hybrid + rerank (`max`, round-1 scoring) | 46.4% | 72.7% | 84.2% | 0.606 | 23.9% | 38.8% | 45.9% | 0.318 | 62.7% |

Files: `results/v2/final/`, `results/v2/final_max_scoring/`.

### App split (round-2 code, for comparison with round 1)

| Mode | Paper Hit@5 / MRR | Corpus Hit@5 / MRR |
|---|---|---|
| Dense | 72.9% / 0.468 | 34.7% / 0.214 |
| Hybrid | 75.7% / 0.480 | 36.1% / 0.240 |
| Hybrid + rerank (`maxp`) | 84.7% / 0.565 | 48.6% / 0.304 |

Round 1 on the same split: 84.7% / 0.564 and 47.9% / 0.304. Files: `results/v2/app/`.
Rerunning the round-1 code unchanged reproduced every round-1 number except one question's
rank in corpus scope (Hit@3 37.5% vs 36.8%): `results/v2/repro_app/`.

### Dev split: what was tried (109 questions)

One question is 0.9 points. Differences of 1-2 questions are noise at this size.

| Variant | Paper Hit@5 / MRR | Corpus Hit@5 / MRR | Retrieval p50, corpus (`bench.py`) | Decision |
|---|---|---|---|---|
| Round-1 config, Chroma HNSW dense search | 89.9% / 0.621 | 51.4% / 0.286 | - | baseline |
| + exact dense search | 89.9% / 0.621 | 50.5% / 0.284 | - | **adopted** (determinism, see below) |
| + rerank depth 50 | 89.9% / 0.621 | 52.3% / 0.290 | 5.23 s vs 3.61 s | rejected: +2 questions for +45% latency |
| + `maxp` scoring | 89.0% / 0.622 | 49.5% / 0.281 | 1.82 s vs 3.61 s | **adopted**: half the latency, −1 question each scope |
| + `maxp`, reranker inputs capped at 256 tokens | 89.0% / 0.622 | 49.5% / 0.281 | 1.68 s | rejected: +7% speed, would truncate long one-paragraph chunks |

First-stage experiments (no reranker, corpus scope, Hit@5; run before the switch to exact
dense search, so absolute numbers can differ from later runs by a question):

| Variant | All | Specific Qs | Generic Qs | Right paper@5 | Decision |
|---|---|---|---|---|---|
| Hybrid, equal RRF weights | 36.7% | 62.5% | 26.0% | 50.5% | kept |
| Dense-heavy RRF (1, 0.5) | 34.9% | 59.4% | 24.7% | 48.6% | rejected |
| BM25-heavy RRF (0.5, 1) | 34.9% | 59.4% | 24.7% | 52.3% | rejected |
| Paper boost, sum of chunk scores (0.5) | 31.2% | 65.6% | 16.9% | 41.3% | rejected |
| Paper boost, title+abstract match (0.5 / 1 / 2) | 33.9 / 35.8 / 33.0% | 68.8% | 19.5 / 22.1 / 18.2% | 48.6 / 45.9 / 40.4% | rejected |

The paper-boost idea (re-rank chunks by how well their paper matches) helps specific
questions but hurts generic ones more, because it confidently pulls in chunks from one wrong
paper. It's still in the code (`PAPER_BOOST`, default off) so the result can be reproduced.

### Why corpus-wide search is so much worse

`failure_analysis.py` assigns each corpus-scope miss one cause (dev: round-1 reranker config
before the exact-search switch; final: the shipped system):

| Cause | Dev (109) | Final (209) |
|---|---|---|
| Hit | 56 | 94 |
| Wrong paper: no top-5 chunk from the asker's paper | 37 (all generic) | 79 (74 generic) |
| Right paper, but gold chunk not among the 30 rerank candidates | 5 | 11 |
| Gold chunk was a candidate, ranked below 5 | 11 | 25 |
| Unreachable (chunk boundaries) | 0 | 0 |

Each question was labelled **specific** (it names something that narrows it to about two of
the 50 papers, like "UTCNN" or "Nepali NER") or **generic** ("What baselines do they
compare to?"). The labels were made by reading only the questions and paper titles, before
any final-split output was seen (`eval/question_specificity_*.json`; one labeller, Claude,
not human-reviewed).

| Final split, corpus scope, Hit@5 | n | Hit@5 |
|---|---|---|
| Specific questions | 90 | 74.4% |
| Generic questions | 119 | 22.7% |

So the gap is mostly not a retrieval defect: QASPER questions were written by people looking
at one paper, and most of them don't say which paper. No ranking change can recover the
paper from "What was the baseline?". Other checks:
- **Chunk boundaries** cause no misses: every gold paragraph is claimed by some chunk.
- **Retrieval depth** has little headroom: gold-in-top-30 is 68% corpus-wide on final,
  gold-in-top-100 is 80% (`results/v2/final/candidate_recall.csv`).
- **Terminology:** misses where the gold chunk never became a candidate have lower word
  overlap with their evidence (median 0.33 vs 0.55 for hits), consistent with vocabulary
  mismatch. It's a small group (11 of 209).

### The product fix: ask "which paper?"

Instead of answering a generic corpus-wide question from whatever paper ranks first (the
right one only 19% of the time on dev), the app now replies "this question doesn't say which
paper it's about" and offers the top three distinct papers as one-click choices. A question
is "specific" when its rarest word appears in at most 6% of the papers (threshold chosen on
dev). Questions that explicitly mention several papers are exempt.

| | Dev | Final |
|---|---|---|
| Agreement with the specificity labels | 85.3% | 82.3% |
| Generic questions caught | 68/77 | 108/119 |
| Specific questions wrongly asked "which paper?" | 7/32 | **26/90** |
| Hit@5 on questions it lets through | 64.7% | 70.7% (vs 45.0% on all) |
| Right paper among the 3 suggestions, when it asks | 39/75 | 54/134 |

The cost: on final, 29% of specific questions get an unnecessary "which paper?" (one click
to resolve). `CLARIFY_GENERIC=0` turns it off. File: `results/v2/clarify_detector.json`.

### Determinism fix

Two processes running the same dense queries against the same Chroma index got different
top-50 lists for 59 of 109 dev questions (approximate HNSW search), which made evaluation
results drift by a question or two between identical runs. Dense search now does an exact
matrix-vector product over the stored vectors. At ~1k chunks per index this is faster than
HNSW (about 20 ms per query including the query embedding), and
`scripts/check_determinism.py` reports 0 differing lists. The original split-level numbers
were barely affected (dev corpus Hit@5 51.4% vs 50.5%).

## 3. Latency

`bench.py`, dev index (926 chunks), first 60 questions, models loaded, one query at a time,
otherwise idle 2-core Intel Xeon @ 2.10 GHz, 2 torch threads, CPU only.

| Stage (ms) | `max` p50 / p95 | `maxp` (shipped) p50 / p95 |
|---|---|---|
| Query embedding + exact dense search | 20 / 23 (paper), 20 / 30 (corpus) | same |
| BM25 | 1 / 2 | same |
| Fusion | < 0.1 | same |
| Rerank, paper scope | 2,526 / 3,602 (55 pairs) | 1,092 / 1,900 (43 pairs) |
| Rerank, corpus scope | 3,587 / 4,308 (96 pairs) | 1,802 / 2,438 (74 pairs) |
| **Total retrieval, paper scope** | 2,547 / 3,632 | **1,113 / 1,920** |
| **Total retrieval, corpus scope** | 3,610 / 4,334 | **1,824 / 2,458** |

Generation, measured separately: qwen3:8b (Q4_K_M) in Ollama on the owner's MacBook Pro,
37 final-set answers sent one at a time, about 1.0-2.3k prompt tokens each:

| | p50 | p95 |
|---|---|---|
| Time to first token | 7.4 s | 10.2 s |
| Total generation time | 9.7 s | 14.4 s |

Generation dominates end-to-end time (about 11-12 s median per answer with corpus-wide
retrieval), and most of it is prompt processing before the first token. Reranking was the
part worth optimising on the server side, and `maxp` halves it for the quality cost shown in
section 2 (final split: −0.013 MRR in both scopes, Hit@5 +1.0 / −0.9 points).

## 4. Answers: correctness, claim support, refusals

### Why the round-1 "100% citation accuracy" wasn't enough

Round 1 measured the share of `[n]` markers that pointed at a retrieved passage. On the
round-2 final set that metric is again **100% (62/62 markers)**. But by manual review only
**80.4%** of cited claims are actually supported by the passages they cite, and four answers
to unanswerable questions assert things no passage says, all with valid-looking markers.

### The evaluation set

`eval/answer_set_{dev,final}.jsonl`. Dev was used to look at failures and choose between
prompts; final was run once.

| Category | Dev | Final | Expected behaviour | Source |
|---|---|---|---|---|
| Answerable | 16 | 24 | correct, cited answer | random sample (seed 7 / 11) of the split's QASPER questions, paper filter on |
| Unanswerable | 4 | 10 | refuse, or say only what the passages support | every QASPER question all annotators marked unanswerable |
| Off-topic | 3 | 4 | refuse | written for this set |
| Ambiguous | 3 | 4 | ask which paper, or attribute | generic questions with no paper picked |
| Multi-paper | 2 | 4 | cover each relevant paper, attributed | written for this set; gold answers checked against the paper text |

Reference review: every sampled answerable question's references were read. Two with
contradictory references (Yes vs No) are kept for claim support but excluded from
correctness, and weak or disagreeing references are noted per item (`review_note`).

### Metrics

- **Answer correctness** (answerable questions with usable references): CORRECT / PARTIAL /
  INCORRECT against any reference; a refusal counts as not correct.
- **Claim support:** the answer is split into sentences; each sentence that cites something
  is a claim, judged SUPPORTED / PARTIAL / UNSUPPORTED against the passages it cites (and
  only those).
- **Citation coverage:** share of sentences that cite anything.
- **Refusal / behaviour:** per category, as in the table above.

### Final-set results (qwen3:8b, prompt v2, shipped retrieval)

| Metric | Value |
|---|---|
| Answerable: correct | **60.9%** (14/23) |
| Answerable: correct or partially correct | 87.0% (20/23) |
| Answerable: refused | 1/24 (false refusal: "Which languages are explored?") |
| Cited claims supported by their passages | **80.4%** (41/51); supported or partly: 90.2% |
| Citation coverage (sentences with a citation) | 82.3% (51/62) |
| Marker validity (round-1 metric) | 100% (62/62) |
| Unanswerable: refused | 3/10 |
| Unanswerable: acceptable (refused, or only passage-supported statements) | 6/10; the other 4 assert unsupported facts |
| Off-topic: refused | 4/4 (all by the relevance floor, before the LLM) |
| Ambiguous: asked "which paper?" | 4/4 |
| Multi-paper: covered both papers with attribution | **0/4** |
| Token F1 vs references (round-1 metric, for continuity) | 0.354 |

All labels above are the reviewer's (`eval/review_final.json`, rubric inside;
`results/v2/answers/final/review_sheet.md` shows every answer next to its references and
cited passages, so anyone can audit or redo the labels).

**Observed failure modes**

1. **Multi-paper questions fail at retrieval.** In 3 of 4, all five retrieved passages came
   from a single paper, so the other paper never reached the model. In the 4th, the second
   paper's passage was retrieved but ignored. A fix would diversify the top-k for
   explicitly cross-paper questions (e.g. at most 2 chunks per paper, k=8). Not implemented
   or measured in this round.
2. **Over-answering unanswerable questions** (4/10): the model writes a plausible answer
   with a valid citation to a passage that doesn't contain it ("The model has one LSTM
   layer [3]").
3. **Right answer, wrong support:** e.g. "English tweets [1]" is correct, but passage [1]
   never mentions English.

### What was tried on generation (dev only)

Prompt v3 targeted failures seen on the dev set: unattributed multi-paper answers, vague
answers to "how much" questions without a number, and copied `BIBREF` markers. It did not
fix them. Unanswerable refusals stayed at 0/4, multi-paper stayed 0/2, `BIBREF` markers were
still copied, and one previously correct answer flipped to a wrong one (v2 correct 66.7% →
v3 60.0% of dev answerable). v2 stays the default; v3 remains selectable
(`PROMPT_VERSION=v3`). What did ship from this analysis: `BIBREF`/`TABREF`-style markers are
stripped when answers are rendered, and generic corpus-wide questions get "which paper?"
instead of an answer. Dev runs: `results/v2/answers/dev_v2`, `dev_v3`; reviews:
`eval/review_dev_*.json`.

### The automated judge and its limits

The planned judge was gemma4:26b (a different model family from the generator). It ran
about 2 minutes per call on the Mac (mostly CPU), so it was replaced by **qwen3:14b**, which
is from the **same family as the generator** and may favour its outputs. Prompts are in
`answer_eval.py` (`JUDGE_SUPPORT`, `JUDGE_CORRECT`). Agreement with the reviewer on the final
set:

| | n | Exact agreement | Binary agreement |
|---|---|---|---|
| Claim support | 51 | 86.3% | 88.2% (supported vs not) |
| Answer correctness | 22 | 68.2% | 72.7% (correct vs not) |

- On claims the judge is slightly **more lenient** (84.3% supported vs 80.4%). It said
  SUPPORTED for 4 claims the reviewer didn't, including the clearest hallucination in the
  set (una00, "cloze questions can be transformed into naturally-looking questions", which
  neither cited passage says).
- On correctness it is much **stricter** (34.8% correct vs 60.9%). It marks answers PARTIAL
  for adding true detail, despite being told not to.
- So the judge is usable as a cheap regression signal for claim support, not as a
  replacement for review, and its correctness scores shouldn't be quoted.
- **The reviewer is also an AI:** all "manual" labels in round 2 were made by Claude in a
  single pass, reading the full passages. When comparing with the judge, the reviewer found
  and fixed 2 of its own labels that had been made from truncated passages. A human
  spot-check of `review_sheet.md` would make these numbers stronger.

### How the answers were generated

The sandbox that ran the evaluation couldn't reach Ollama on the Mac, so a browser tab on
the Mac ran the prompts (`eval/relay_runner.js`). Each prompt was rebuilt in the browser
from the exact text Python built, and its SHA-1 was checked on both sides. All 79 answer
prompts (dev v2, dev v3, final) and 77 judge prompts matched. With direct access to Ollama,
`answer_eval.py generate` does the same in one step.

## 5. Limitations

- Small answer sets (24 answerable on final): one answer is about 4 points. Treat the answer
  metrics as indicative.
- One reviewer, an AI, no inter-annotator agreement.
- Retrieval scores depend on QASPER's evidence annotations, which are incomplete. A
  "miss" can be a different paragraph that also answers the question.
- The "specific vs generic" labels are subjective, and so is the 6% threshold.
- Latency was measured on a 2-core cloud VM (retrieval) and one laptop (generation);
  absolute numbers will differ elsewhere, but the comparison between settings holds.
