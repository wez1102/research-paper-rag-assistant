"""
Ablation study: which retrieval tricks actually move Hit@k / MRR?

The danger with tuning on a 144-question benchmark is that you end up tuning *to*
those 144 questions. So this script uses two disjoint sets of QASPER papers:

  dev   - validation papers 50-99 (~150 questions). Every setting is chosen here.
  test  - validation papers 0-49, the same 144 questions as evaluate.py. Only
          looked at after the choice is made, to report the final numbers.

Ideas being tested (all with the same two small models, so gains come from method):

  MaxP ("max passage")   A 450-token chunk is a lot of text for one embedding or one
                         cross-encoder pass; a single relevant sentence gets diluted.
                         So we also split each chunk into paragraph-sized passages,
                         score those, and give each chunk the score of its best passage.
  Fusion variants        Mix chunk-level and passage-level dense/BM25 rankings with
                         (weighted) reciprocal rank fusion.
  Rerank depth           How many fused candidates the cross-encoder looks at.
  Prior fusion           Blend the cross-encoder ranking back with the first-stage
                         ranking, so a noisy cross-encoder score can't bury a chunk
                         that both retrievers agreed on.

Everything slow (embeddings, cross-encoder scores) is cached in data/cache/, so the
run can be interrupted and resumed:

    python experiments.py                 # does everything, then writes the report
    python experiments.py --budget 150    # stop after ~150 s, re-run to continue
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import settings
from ingest import build_questions, chunk_paper, embedding_text, load_papers, paper_paragraphs
from retrieve import split_passages, tokenize

CACHE = settings.data_dir / "cache"


def _slug(model: str) -> str:
    return model.rstrip("/").split("/")[-1]


# Cache files carry the model names, so trying a different model never mixes scores.
EMB_TAG = _slug(settings.embed_model)
CE_TAG = f"{EMB_TAG}__{_slug(settings.rerank_model)}"
DEFAULT_MODELS = (settings.embed_model.endswith("bge-small-en-v1.5")
                  and settings.rerank_model.endswith("ms-marco-MiniLM-L-6-v2"))
# Results for non-default models get a suffix, e.g. results/ablation__bge-base-en-v1.5__bge-reranker-base.md
OUT_SUFFIX = "" if DEFAULT_MODELS else f"__{CE_TAG}"
SPLITS = {"dev": 50, "test": 0}          # first QASPER row of each 50-paper slice
SCOPES = ("corpus", "paper")
POOL = 50                                 # depth of each first-stage ranking
CE_DEPTH = 30                             # max candidates we ever send to the cross-encoder


class OutOfTime(Exception):
    """Raised when --budget runs out; everything computed so far is already on disk."""


# ---------------------------------------------------------------------------
# Building the two splits
# ---------------------------------------------------------------------------

@dataclass
class Split:
    name: str
    chunks: list            # ingest.Chunk objects
    passages: list          # (chunk index, passage text)
    questions: list         # dicts from ingest.build_questions

    def chunk_texts(self) -> list[str]:
        return [embedding_text(c.title, c.section, c.text) for c in self.chunks]

    def passage_texts(self) -> list[str]:
        return [embedding_text(self.chunks[ci].title, self.chunks[ci].section, p) for ci, p in self.passages]


def build_split(name: str, tokenizer) -> Split:
    chunks, questions = [], []
    for paper in load_papers(50, SPLITS[name]):
        paras = paper_paragraphs(paper)
        chunks.extend(chunk_paper(paras, tokenizer, settings.chunk_size, settings.chunk_overlap))
        questions.extend(build_questions(paper, paras)[0])
    passages = [(ci, p) for ci, c in enumerate(chunks) for p in split_passages(c.text)]
    return Split(name, chunks, passages, questions)


# ---------------------------------------------------------------------------
# Cached, resumable model calls
# ---------------------------------------------------------------------------

def _fingerprint(texts: list[str]) -> str:
    return hashlib.sha1("\x00".join(texts).encode()).hexdigest()[:16]


def embed_cached(tag: str, texts: list[str], embedder, deadline: float) -> np.ndarray:
    """Embed `texts`, saving progress every batch. A changed corpus invalidates the cache."""
    path, fp_path = CACHE / f"emb_{EMB_TAG}_{tag}.npy", CACHE / f"emb_{EMB_TAG}_{tag}.fp"
    fp = _fingerprint(texts)
    done = np.zeros((0, embedder.get_sentence_embedding_dimension()), dtype=np.float32)
    if path.exists() and fp_path.exists() and fp_path.read_text() == fp:
        done = np.load(path)
    fp_path.write_text(fp)

    while len(done) < len(texts):
        if time.time() > deadline:
            raise OutOfTime(f"embeddings for {tag}: {len(done)}/{len(texts)}")
        batch = texts[len(done):len(done) + 128]
        vecs = embedder.encode(batch, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        done = np.vstack([done, np.asarray(vecs, dtype=np.float32)])
        np.save(path, done)
    return done


class CECache:
    """Cross-encoder scores keyed by 'question_id|unit', persisted as JSON."""

    def __init__(self, tag: str):
        self.path = CACHE / f"ce_{CE_TAG}_{tag}.json"
        self.scores: dict[str, float] = json.loads(self.path.read_text()) if self.path.exists() else {}

    def fill(self, pairs: dict[str, tuple[str, str]], model, deadline: float) -> None:
        todo = [k for k in pairs if k not in self.scores]
        for i in range(0, len(todo), 64):
            if time.time() > deadline:
                self.save()
                raise OutOfTime(f"cross-encoder {self.path.name}: {len(self.scores)} scored, {len(todo) - i} to go")
            keys = todo[i:i + 64]
            preds = model.predict([pairs[k] for k in keys], batch_size=32, show_progress_bar=False)
            self.scores.update({k: float(s) for k, s in zip(keys, preds)})
            if i % 640 == 0:
                self.save()
        self.save()

    def save(self) -> None:
        self.path.write_text(json.dumps(self.scores))


# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------

def wrrf(lists: list[list[int]], weights: list[float] | None = None, k: int = 60) -> list[int]:
    """Weighted reciprocal rank fusion over lists of chunk indices."""
    weights = weights or [1.0] * len(lists)
    score: dict[int, float] = defaultdict(float)
    for lst, w in zip(lists, weights):
        for rank, item in enumerate(lst, start=1):
            score[item] += w / (k + rank)
    return sorted(score, key=score.get, reverse=True)


def top(scores: np.ndarray, cand: np.ndarray, n: int = POOL, positive_only: bool = False) -> list[int]:
    s = scores[cand]
    order = np.argsort(-s, kind="stable")
    if positive_only:
        order = order[s[order] > 0]
    return [int(cand[i]) for i in order[:n]]


def max_per_chunk(passage_scores: np.ndarray, passage_chunk: np.ndarray, n_chunks: int) -> np.ndarray:
    out = np.full(n_chunks, -np.inf)
    np.maximum.at(out, passage_chunk, passage_scores)
    return out


class Ranker:
    """Holds one split's embeddings / BM25 and produces every ranking variant per question."""

    def __init__(self, split: Split, chunk_emb, pass_emb, q_emb):
        from rank_bm25 import BM25Okapi

        self.s = split
        self.chunk_emb, self.pass_emb, self.q_emb = chunk_emb, pass_emb, q_emb
        self.chunk_paper = np.array([c.paper_id for c in split.chunks])
        self.pass_chunk = np.array([ci for ci, _ in split.passages])
        self.passages_of: dict[int, list[int]] = defaultdict(list)
        for pi, ci in enumerate(self.pass_chunk):
            self.passages_of[int(ci)].append(pi)
        self.bm25_chunk = BM25Okapi([tokenize(t) for t in split.chunk_texts()])
        self.bm25_pass = BM25Okapi([tokenize(t) for t in split.passage_texts()])
        self._first_stage: dict[tuple[int, str], dict[str, list[int]]] = {}

    def candidates(self, qi: int, scope: str) -> np.ndarray:
        if scope == "paper":
            return np.flatnonzero(self.chunk_paper == self.s.questions[qi]["paper_id"])
        return np.arange(len(self.s.chunks))

    def first_stage(self, qi: int, scope: str) -> dict[str, list[int]]:
        key = (qi, scope)
        if key in self._first_stage:
            return self._first_stage[key]
        n = len(self.s.chunks)
        cand = self.candidates(qi, scope)
        q = self.q_emb[qi]
        toks = tokenize(self.s.questions[qi]["question"])

        dense = top(self.chunk_emb @ q, cand)
        dense_maxp = top(max_per_chunk(self.pass_emb @ q, self.pass_chunk, n), cand)
        bm25 = top(self.bm25_chunk.get_scores(toks), cand, positive_only=True)
        bm25_maxp = top(max_per_chunk(self.bm25_pass.get_scores(toks), self.pass_chunk, n), cand,
                        positive_only=True)

        out = {
            "dense": dense,
            "dense_maxp": dense_maxp,
            "bm25": bm25,
            "hybrid": wrrf([dense, bm25])[:POOL],                              # what's in production today
            "dense+maxp": wrrf([dense, dense_maxp])[:POOL],
            "hybrid_maxp": wrrf([dense, dense_maxp, bm25, bm25_maxp])[:POOL],
            "hybrid_maxp_dense_heavy": wrrf([dense, dense_maxp, bm25, bm25_maxp], [1, 1, .5, .5])[:POOL],
        }
        self._first_stage[key] = out
        return out

    def ce_pairs(self) -> dict[str, tuple[str, str]]:
        """Every (question, text) pair any rerank variant could need."""
        chunk_txt, pass_txt = self.s.chunk_texts(), self.s.passage_texts()
        pairs = {}
        for qi, q in enumerate(self.s.questions):
            for scope in SCOPES:
                fs = self.first_stage(qi, scope)
                for ci in set(fs["hybrid"][:CE_DEPTH]) | set(fs["hybrid_maxp"][:CE_DEPTH]):
                    pairs[f"{q['question_id']}|c{ci}"] = (q["question"], chunk_txt[ci])
                    for pi in self.passages_of[ci]:
                        pairs[f"{q['question_id']}|p{pi}"] = (q["question"], pass_txt[pi])
        return pairs

    def reranked(self, qi: int, base: list[int], depth: int, how: str, prior: float,
                 ce: dict[str, float]) -> list[int]:
        qid = self.s.questions[qi]["question_id"]
        head, tail = base[:depth], base[depth:]

        def ce_chunk(ci):
            return ce[f"{qid}|c{ci}"]

        def ce_maxp(ci):
            return max(ce[f"{qid}|p{pi}"] for pi in self.passages_of[ci])

        score = {"chunk": ce_chunk, "maxp": ce_maxp,
                 "max": lambda ci: max(ce_chunk(ci), ce_maxp(ci)),
                 "mean": lambda ci: (ce_chunk(ci) + ce_maxp(ci)) / 2}[how]
        order = sorted(head, key=score, reverse=True)
        if prior > 0:
            # Blend with the first-stage order: a chunk ranked highly by both survives.
            order = wrrf([order, head], [1.0, prior])
        return order + tail


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def metrics(split: Split, ranked: list[list[int]]) -> dict[str, float]:
    ranks = []
    for q, lst in zip(split.questions, ranked):
        gold, rank = set(q["gold_paragraph_ids"]), None
        for r, ci in enumerate(lst[:5], start=1):
            c = split.chunks[ci]
            if c.paper_id == q["paper_id"] and gold & set(c.paragraph_ids):
                rank = r
                break
        ranks.append(rank)
    n = len(ranks)
    out = {f"hit@{k}": sum(r is not None and r <= k for r in ranks) / n for k in (1, 3, 5)}
    out["mrr"] = sum(1 / r for r in ranks if r) / n
    return out


def all_variants(ranker: Ranker, ce: dict[str, float]):
    """Yield (variant_name, config_dict, {scope: ranked lists})."""
    s = ranker.s
    fs = {scope: [ranker.first_stage(qi, scope) for qi in range(len(s.questions))] for scope in SCOPES}
    for name in fs["paper"][0]:
        yield name, {"stage": "first", "list": name}, {sc: [f[name] for f in fs[sc]] for sc in SCOPES}
    for base in ("hybrid", "hybrid_maxp"):
        for depth in (10, 20, 30):
            for how in ("chunk", "maxp", "max", "mean"):
                for prior in (0.0, 0.25, 0.5, 1.0):
                    cfg = {"stage": "rerank", "base": base, "depth": depth, "how": how, "prior": prior}
                    name = f"rerank[{base},d={depth},{how},prior={prior}]"
                    yield name, cfg, {sc: [ranker.reranked(qi, fs[sc][qi][base], depth, how, prior, ce)
                                          for qi in range(len(s.questions))] for sc in SCOPES}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval ablation study (dev-tuned, test-reported).")
    parser.add_argument("--budget", type=float, default=0, help="Stop after this many seconds (0 = no limit).")
    args = parser.parse_args()
    deadline = time.time() + args.budget if args.budget else float("inf")
    CACHE.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import CrossEncoder, SentenceTransformer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()

    embedder = SentenceTransformer(settings.embed_model, device=settings.device)
    splits = {name: build_split(name, embedder.tokenizer) for name in SPLITS}
    for s in splits.values():
        print(f"{s.name}: {len(s.chunks)} chunks, {len(s.passages)} passages, {len(s.questions)} questions")

    try:
        rankers = {}
        for name, s in splits.items():
            q_emb = embed_cached(f"{name}_q", [settings.query_instruction + q["question"] for q in s.questions],
                                 embedder, deadline)
            c_emb = embed_cached(f"{name}_chunks", s.chunk_texts(), embedder, deadline)
            p_emb = embed_cached(f"{name}_passages", s.passage_texts(), embedder, deadline)
            rankers[name] = Ranker(s, c_emb, p_emb, q_emb)

        reranker = CrossEncoder(settings.rerank_model, device=settings.device, max_length=512)
        ce = {}
        for name in ("dev", "test"):
            cache = CECache(name)
            cache.fill(rankers[name].ce_pairs(), reranker, deadline)
            ce[name] = cache.scores
    except OutOfTime as stop:
        print(f"Paused ({stop}). Re-run to continue - nothing is lost.")
        return

    # --- score every variant on both splits ----------------------------------
    rows = []
    for split_name, ranker in rankers.items():
        for name, cfg, ranked in all_variants(ranker, ce[split_name]):
            for scope in SCOPES:
                rows.append({"split": split_name, "variant": name, "scope": scope, **cfg,
                             **metrics(ranker.s, ranked[scope])})
    df = pd.DataFrame(rows)
    settings.results_dir.mkdir(exist_ok=True)
    df.to_csv(settings.results_dir / f"ablation_all{OUT_SUFFIX}.csv", index=False)

    # --- pick on dev only ------------------------------------------------------
    dev = df[df.split == "dev"].pivot_table(index="variant", columns="scope", values="mrr")
    dev["mean_mrr"] = dev.mean(axis=1)
    first = [v for v in dev.index if not v.startswith("rerank")]
    best_first = dev.loc[first, "mean_mrr"].idxmax()
    best_rerank = dev.drop(index=first)["mean_mrr"].idxmax()
    picks = {
        "Dense (baseline)": "dense",
        "Hybrid (baseline)": "hybrid",
        "Hybrid + rerank (baseline)": "rerank[hybrid,d=20,chunk,prior=0.0]",
        "Best first stage (dev-picked)": best_first,
        "Best reranker (dev-picked)": best_rerank,
    }
    summary = []
    for label, variant in picks.items():
        for split_name in ("dev", "test"):
            for scope in SCOPES:
                r = df[(df.split == split_name) & (df.variant == variant) & (df.scope == scope)].iloc[0]
                summary.append({"label": label, "variant": variant, "split": split_name, "scope": scope,
                                **{m: r[m] for m in ("hit@1", "hit@3", "hit@5", "mrr")}})
    summ = pd.DataFrame(summary)
    summ.to_csv(settings.results_dir / f"ablation{OUT_SUFFIX}.csv", index=False)

    best_cfg = df[df.variant == best_rerank].iloc[0][["base", "depth", "how", "prior"]].to_dict()
    (settings.results_dir / f"ablation_best{OUT_SUFFIX}.json").write_text(json.dumps(
        {"best_first_stage": best_first, "best_rerank": best_rerank, "best_rerank_config": best_cfg},
        indent=2, default=str))

    with (settings.results_dir / f"ablation{OUT_SUFFIX}.md").open("w") as f:
        for split_name in ("dev", "test"):
            for scope in SCOPES:
                f.write(f"### {split_name} / {scope}\n\n| Setting | Variant | Hit@1 | Hit@3 | Hit@5 | MRR |\n"
                        "|---|---|---|---|---|---|\n")
                for _, r in summ[(summ.split == split_name) & (summ.scope == scope)].iterrows():
                    f.write(f"| {r['label']} | `{r['variant']}` | {r['hit@1']:.1%} | {r['hit@3']:.1%} | "
                            f"{r['hit@5']:.1%} | {r['mrr']:.3f} |\n")
                f.write("\n")
    print((settings.results_dir / f"ablation{OUT_SUFFIX}.md").read_text())
    print("DONE. Best (picked on dev):", best_first, "|", best_rerank)


if __name__ == "__main__":
    main()
