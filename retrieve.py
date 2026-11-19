"""
Step 2: find the chunks that are most likely to answer a question.

Three retrieval modes, all behind one `Retriever.retrieve()` call so the app and
the evaluation script use exactly the same code path:

  dense          - embed the question with bge-small, nearest neighbours in Chroma.
  hybrid         - dense + BM25 keyword search, merged with reciprocal rank fusion.
  hybrid_rerank  - take the top 30 hybrid results and let a cross-encoder re-score
                   them (each chunk is scored by its best paragraph - "MaxP"),
                   blend that with the hybrid order, keep 5.

Why bother with BM25? Embeddings are great at paraphrases but fuzzy on exact
tokens: model names ("BERT-large"), dataset names ("SQuAD 2.0"), metric names.
Keyword search nails those, and fusion gets us the best of both.

Quick check from the terminal:
    python retrieve.py "Which datasets are used for evaluation?" --mode hybrid
"""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from dataclasses import dataclass, field

import numpy as np

from config import settings
from ingest import embedding_text, model_tag

MODES = ("dense", "hybrid", "hybrid_rerank")

# A short stopword list is enough for BM25 here - we mostly want to stop "the" and
# "of" from dominating scores, not do linguistics.
STOPWORDS = set("""
a an and are as at be by did do does for from has have how in is it its of on or
that the their them they this to was were what when where which who why will with
we our you your can could should would these those than then there into also such
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens for BM25. Keeps things like 'bert-base' and 'f1' intact."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60,
                           weights: list[float] | None = None) -> list[tuple[str, float]]:
    """
    Merge several ranked lists of ids. Each list gives an item weight / (k + rank) points.

    RRF only looks at ranks, not raw scores, which is exactly what we want: cosine
    similarities and BM25 scores live on completely different scales, and
    normalising them against each other is fiddly and brittle.
    """
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def split_passages(text: str, max_words: int = 160, window: int = 120, stride: int = 90,
                   min_words: int = 25) -> list[str]:
    """
    Cut a chunk into paragraph-sized passages for the reranker.

    Why: a 450-token chunk often holds one relevant sentence and a lot of filler,
    and a cross-encoder (trained on short MS MARCO passages) scores that mix poorly.
    Scoring each paragraph separately and keeping the best one ("MaxP") fixes that.
    Chunks keep paragraph breaks ("\\n\\n"), so we split on those, glue tiny
    fragments onto a neighbour (a 10-word passage gives a noisy score), and window
    anything very long.
    """
    merged: list[str] = []
    for piece in (p.strip() for p in text.split("\n\n")):
        if not piece:
            continue
        if merged and len(merged[-1].split()) < min_words:
            merged[-1] = merged[-1] + " " + piece
        else:
            merged.append(piece)
    if len(merged) > 1 and len(merged[-1].split()) < min_words:
        tail = merged.pop()
        merged[-1] = merged[-1] + " " + tail

    out: list[str] = []
    for p in merged:
        words = p.split()
        if len(words) <= max_words:
            out.append(p)
            continue
        for s in range(0, len(words), stride):
            out.append(" ".join(words[s:s + window]))
            if s + window >= len(words):
                break
    return out


@dataclass
class RetrievedChunk:
    chunk_id: str
    paper_id: str
    title: str
    section: str
    text: str
    paragraph_ids: list[int]
    score: float
    # Where the score came from ("dense", "rrf", "rerank") - shown in the UI.
    score_type: str = ""
    extra: dict = field(default_factory=dict)


class IndexNotReady(RuntimeError):
    """The index is missing, empty, or was built with a different embedding model."""


class Retriever:
    """Loads the index once and answers retrieval queries in any of the three modes."""

    def __init__(self, collection_name: str | None = None, load_reranker: bool = True,
                 score_cache: dict | None = None):
        import chromadb
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        self.collection_name = collection_name or settings.collection_name()
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        try:
            self.collection = client.get_collection(self.collection_name)
        except Exception as exc:
            raise IndexNotReady(
                f"Collection '{self.collection_name}' not found in {settings.chroma_dir}. "
                "Run `python ingest.py` first."
            ) from exc

        # Catch the quiet failure modes early: an empty collection, or one embedded with
        # a different model than the one we'd use for queries (the vectors would be
        # meaningless to each other, and nothing would error - results would just be bad).
        self.manifest = dict(self.collection.metadata or {})
        count = self.collection.count()
        if count == 0:
            raise IndexNotReady(f"Collection '{self.collection_name}' is empty. Run `python ingest.py --force`.")
        built_with = self.manifest.get("embed_model")
        if built_with and built_with != model_tag(settings.embed_model):
            raise IndexNotReady(f"Index was built with '{built_with}' but EMBED_MODEL is "
                                f"'{model_tag(settings.embed_model)}'. Re-run `python ingest.py`.")
        if self.manifest.get("n_chunks") not in (None, count):
            raise IndexNotReady(f"Index holds {count} records but its manifest says {self.manifest['n_chunks']}. "
                                "Re-run `python ingest.py --force`.")

        # Pull every chunk into memory. With ~1-2k chunks this is a few MB and it
        # lets BM25 and the reranker work without extra round-trips to Chroma.
        # Sorted by id so every process sees the same order (ties break the same way).
        dump = self.collection.get(include=["documents", "metadatas", "embeddings"])
        order = sorted(range(len(dump["ids"])), key=lambda i: dump["ids"][i])
        self.ids: list[str] = [dump["ids"][i] for i in order]
        self.chunks: dict[str, dict] = {}
        for i in order:
            self.chunks[dump["ids"][i]] = {"text": dump["documents"][i], **dump["metadatas"][i]}
        self._paper_of = np.array([self.chunks[c]["paper_id"] for c in self.ids])
        # Dense search is exact (one matrix-vector product) rather than Chroma's
        # approximate HNSW search. At this size it's faster, and it is deterministic:
        # in round 2 we found HNSW returned different top-50 lists from one process to
        # the next for over half the dev questions, which made results drift by a
        # question or two between identical runs. (Past ~1M chunks, go back to ANN.)
        self._vecs = np.asarray([dump["embeddings"][i] for i in order], dtype=np.float32)

        # BM25 indexes the same "title. section. text" string we embedded.
        corpus = [tokenize(embedding_text(self.chunks[c]["title"], self.chunks[c]["section"],
                                          self.chunks[c]["text"])) for c in self.ids]
        self.bm25 = BM25Okapi(corpus)
        # In how many papers does each word occur? Used to tell "What was the baseline?"
        # (words found in nearly every paper) from "What is UTCNN?" (a word one paper owns).
        paper_terms: dict[str, set] = {}
        for cid, toks in zip(self.ids, corpus):
            paper_terms.setdefault(self.chunks[cid]["paper_id"], set()).update(toks)
        self.n_papers = len(paper_terms)
        self.paper_df: dict[str, int] = {}
        for terms in paper_terms.values():
            for t in terms:
                self.paper_df[t] = self.paper_df.get(t, 0) + 1

        # Title + abstract of each paper, for the "abstract" paper prior (see paper_aware).
        # QASPER's question writers saw only these when writing questions, so a question
        # usually echoes its paper's abstract more than any random body passage.
        self.abstract_ids = [c for c in self.ids if self.chunks[c]["section"] == "Abstract"]
        self._abstract_bm25 = BM25Okapi([tokenize(embedding_text(self.chunks[c]["title"], "Abstract",
                                                                 self.chunks[c]["text"]))
                                         for c in self.abstract_ids]) if self.abstract_ids else None
        self._abstract_vecs = None          # fetched from Chroma on first use

        # Passages for MaxP reranking, prepared once up front (it's just string splitting).
        self.passages = {c: split_passages(self.chunks[c]["text"]) for c in self.ids}

        self.embedder = SentenceTransformer(settings.embed_model, device=settings.device)
        self._reranker = None
        # Optional {hash(question, text): score} memo for the cross-encoder. evaluate.py
        # passes one in so that trying several rerank depths doesn't re-score the same
        # pairs; the app leaves it off.
        self.score_cache = score_cache
        if load_reranker:
            self._load_reranker()

    # -- helpers -------------------------------------------------------------

    def question_specificity(self, question: str) -> dict:
        """
        Does the question point at one paper on its own? Looks at its rarest word
        (that occurs in the corpus at all): if that word shows up in at most
        `generic_paper_share` of the papers, the question names something specific.
        On the dev split this agreed with hand labels for 85% of questions.
        """
        counts = {t: self.paper_df[t] for t in tokenize(question) if t in self.paper_df}
        limit = max(1, round(settings.generic_paper_share * self.n_papers))
        if not counts:
            return {"specific": False, "rarest_term": None, "papers_with_term": None, "limit": limit}
        term = min(counts, key=counts.get)
        return {"specific": counts[term] <= limit, "rarest_term": term, "papers_with_term": counts[term],
                "limit": limit}

    def _load_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(settings.rerank_model, device=settings.device,
                                          max_length=settings.rerank_max_length)
        return self._reranker

    def papers(self) -> list[tuple[str, str]]:
        """(paper_id, title) for every paper in the index, sorted by title. Used by the UI filter."""
        seen = {m["paper_id"]: m["title"] for m in self.chunks.values()}
        return sorted(seen.items(), key=lambda kv: kv[1].lower())

    def _to_result(self, cid: str, score: float, score_type: str) -> RetrievedChunk:
        c = self.chunks[cid]
        return RetrievedChunk(
            chunk_id=cid,
            paper_id=c["paper_id"],
            title=c["title"],
            section=c["section"],
            text=c["text"],
            paragraph_ids=[int(x) for x in str(c["paragraph_ids"]).split(",") if x != ""],
            score=float(score),
            score_type=score_type,
        )

    # -- the individual retrievers ------------------------------------------

    def dense_search(self, question: str, n: int, paper_id: str | None = None) -> list[tuple[str, float]]:
        """Nearest neighbours by cosine similarity (exact). Returns (chunk_id, similarity)."""
        q = self.embedder.encode(settings.query_instruction + question, normalize_embeddings=True)
        sims = self._vecs @ np.asarray(q, dtype=np.float32)      # vectors are normalised -> cosine
        if paper_id:
            sims = np.where(self._paper_of == paper_id, sims, -np.inf)
        top = np.argsort(-sims, kind="stable")[:n]
        return [(self.ids[i], float(sims[i])) for i in top if np.isfinite(sims[i])]

    def bm25_search(self, question: str, n: int, paper_id: str | None = None) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(question))
        if paper_id:
            scores = np.where(self._paper_of == paper_id, scores, -np.inf)
        top = np.argsort(-scores, kind="stable")[:n]
        # Drop zero-score hits: they share no words with the question at all.
        return [(self.ids[i], float(scores[i])) for i in top if np.isfinite(scores[i]) and scores[i] > 0]

    def hybrid_search(self, question: str, n: int, paper_id: str | None = None,
                      timings: dict | None = None) -> list[tuple[str, float]]:
        timings = {} if timings is None else timings
        pool = settings.fusion_pool
        t0 = time.perf_counter()
        dense = [cid for cid, _ in self.dense_search(question, pool, paper_id)]
        t1 = time.perf_counter()
        sparse = [cid for cid, _ in self.bm25_search(question, pool, paper_id)]
        t2 = time.perf_counter()
        fused = reciprocal_rank_fusion([dense, sparse], k=settings.rrf_k, weights=list(settings.rrf_weights))
        if paper_id is None and settings.paper_boost > 0:
            fused = self.paper_aware(fused, settings.paper_boost, question)
        timings.update(dense_s=t1 - t0, bm25_s=t2 - t1, fusion_s=time.perf_counter() - t2)
        return fused[:n]

    def abstract_paper_ranking(self, question: str) -> list[str]:
        """Papers ordered by how well their title + abstract match the question (dense + BM25, RRF)."""
        if not self.abstract_ids:
            return []
        if self._abstract_vecs is None:
            pos = {c: i for i, c in enumerate(self.ids)}
            self._abstract_vecs = self._vecs[[pos[c] for c in self.abstract_ids]]
        q = self.embedder.encode(settings.query_instruction + question, normalize_embeddings=True)
        dense = [self.abstract_ids[i] for i in np.argsort(-(self._abstract_vecs @ q), kind="stable")]
        sparse = [self.abstract_ids[i] for i in np.argsort(-self._abstract_bm25.get_scores(tokenize(question)),
                                                           kind="stable")]
        fused = reciprocal_rank_fusion([dense, sparse], k=settings.rrf_k)
        return [self.chunks[c]["paper_id"] for c, _ in fused]

    def paper_aware(self, fused: list[tuple[str, float]], boost: float,
                    question: str = "") -> list[tuple[str, float]]:
        """
        Corpus-wide search only. Re-rank chunks using how well their *paper* matches.

        Why: when no paper is picked, most misses aren't "right paper, wrong passage" -
        the right paper isn't in the top 5 at all. The words of a question are often
        spread over several passages of the paper it's about (the method in one, the
        dataset in another), so a paper with several matching chunks is a better bet
        than one with a single lucky match. A paper's score is the sum of its chunks'
        fused scores in the candidate pool; chunks are then ordered by their paper's
        rank, and that order is fused back with the plain chunk order (weight `boost`).
        """
        chunk_order = [cid for cid, _ in fused]
        if settings.paper_prior == "abstract" and question:
            ranking = self.abstract_paper_ranking(question)
        else:
            paper_score: dict[str, float] = {}
            for cid, score in fused:
                pid = self.chunks[cid]["paper_id"]
                paper_score[pid] = paper_score.get(pid, 0.0) + score
            ranking = sorted(paper_score, key=paper_score.get, reverse=True)
        paper_rank = {pid: r for r, pid in enumerate(ranking)}
        position = {c: i for i, c in enumerate(chunk_order)}
        last = len(paper_rank)
        by_paper = sorted(chunk_order, key=lambda c: (paper_rank.get(self.chunks[c]["paper_id"], last), position[c]))
        return reciprocal_rank_fusion([chunk_order, by_paper], k=settings.rrf_k, weights=[1.0, boost])

    def rerank(self, question: str, candidate_ids: list[str], k: int) -> list[tuple[str, float]]:
        """
        Re-score candidates with the cross-encoder, which reads question and text
        together - slower than embeddings, but much sharper.

        Each chunk is scored twice: as a whole, and paragraph by paragraph. How the
        two are combined (settings.rerank_how) and how much the first-stage order
        still counts (settings.rerank_prior) were picked on a separate dev set by
        experiments.py; "max" + prior 0.5 won there in round 1. Round 2 moved to
        "maxp" for speed (see config.rerank_how).
        """
        if not candidate_ids:
            return []
        model = self._load_reranker()

        # One batched predict() call for every chunk and every passage.
        # With rerank_how="maxp" a multi-paragraph chunk is judged by its paragraphs
        # alone, so its (long, slow) whole-chunk pair is skipped.
        need_chunk = settings.rerank_how != "maxp"
        pairs, owner = [], []
        for c in candidate_ids:
            meta = self.chunks[c]
            single = self.passages[c] == [meta["text"].strip()]
            if need_chunk or single:
                pairs.append((question, embedding_text(meta["title"], meta["section"], meta["text"])))
                owner.append((c, "chunk"))
            # A one-paragraph chunk's only passage is the chunk itself - about a quarter of
            # all chunks. No point scoring the same text twice; the chunk score stands in.
            if single:
                continue
            for p in self.passages[c]:
                pairs.append((question, embedding_text(meta["title"], meta["section"], p)))
                owner.append((c, "passage"))
        scores = self._predict(model, pairs)
        self.last_pairs = len(pairs)

        chunk_score: dict[str, float] = {}
        best_passage: dict[str, float] = {}
        for (c, kind), s in zip(owner, scores):
            if kind == "chunk":
                chunk_score[c] = float(s)
            else:
                best_passage[c] = max(best_passage.get(c, -1e9), float(s))

        how = settings.rerank_how
        combine = {
            "chunk": lambda c: chunk_score[c],
            "maxp": lambda c: best_passage[c] if c in best_passage else chunk_score[c],
            "max": lambda c: max(chunk_score[c], best_passage.get(c, -1e9)),
            "mean": lambda c: (chunk_score[c] + best_passage.get(c, chunk_score[c])) / 2,
        }[how]
        final = {c: combine(c) for c in candidate_ids}
        order = sorted(candidate_ids, key=final.get, reverse=True)

        if settings.rerank_prior > 0:
            # A chunk that both the retrievers and the cross-encoder like should win
            # over one only the cross-encoder likes, so blend the two orders.
            order = [c for c, _ in reciprocal_rank_fusion([order, list(candidate_ids)], k=settings.rrf_k,
                                                          weights=[1.0, settings.rerank_prior])]
        # The reported score stays the cross-encoder's (the refusal check in generate.py uses it).
        return [(c, final[c]) for c in order[:k]]

    def _predict(self, model, pairs: list[tuple[str, str]]) -> list[float]:
        """Cross-encoder scores, served from score_cache where possible."""
        if self.score_cache is None:
            return [float(x) for x in model.predict(pairs, batch_size=32, show_progress_bar=False)]
        keys = [hashlib.sha1(f"{q}\x00{t}".encode()).hexdigest() for q, t in pairs]
        todo = [i for i, key in enumerate(keys) if key not in self.score_cache]
        if todo:
            fresh = model.predict([pairs[i] for i in todo], batch_size=32, show_progress_bar=False)
            for i, s in zip(todo, fresh):
                self.score_cache[keys[i]] = float(s)
        return [self.score_cache[key] for key in keys]

    # -- the one function everyone calls -------------------------------------

    def retrieve(self, question: str, mode: str = "hybrid_rerank", k: int | None = None,
                 paper_id: str | None = None, timings: dict | None = None) -> list[RetrievedChunk]:
        """
        Return the top-k chunks for `question` using the chosen mode. Pass a dict as
        `timings` to get the seconds spent in each stage (dense_s, bm25_s, fusion_s,
        rerank_s) - used by the request logs and bench.py.
        """
        timings = {} if timings is None else timings
        k = k or settings.top_k
        if mode == "dense":
            t0 = time.perf_counter()
            hits, kind = self.dense_search(question, k, paper_id), "dense"
            timings["dense_s"] = time.perf_counter() - t0
        elif mode == "hybrid":
            hits, kind = self.hybrid_search(question, k, paper_id, timings), "rrf"
        elif mode == "hybrid_rerank":
            candidates = [cid for cid, _ in self.hybrid_search(question, settings.rerank_candidates, paper_id,
                                                               timings)]
            t0 = time.perf_counter()
            hits, kind = self.rerank(question, candidates, k), "rerank"
            timings["rerank_s"] = time.perf_counter() - t0
            timings["rerank_pairs"] = getattr(self, "last_pairs", 0)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Pick one of {MODES}.")
        return [self._to_result(cid, score, kind) for cid, score in hits]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single retrieval query.")
    parser.add_argument("question")
    parser.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    parser.add_argument("-k", type=int, default=settings.top_k)
    args = parser.parse_args()

    retriever = Retriever(load_reranker=args.mode == "hybrid_rerank")
    for i, r in enumerate(retriever.retrieve(args.question, args.mode, args.k), start=1):
        print(f"[{i}] {r.score_type}={r.score:.3f}  {r.title}  ::  {r.section}")
        print("    " + r.text[:200].replace("\n", " ") + "...")


if __name__ == "__main__":
    main()
