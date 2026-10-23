"""
Step 2: find the chunks that are most likely to answer a question.

Three retrieval modes, all behind one `Retriever.retrieve()` call so the app and
the evaluation script use exactly the same code path:

  dense          - embed the question with bge-small, nearest neighbours in Chroma.
  hybrid         - dense + BM25 keyword search, merged with reciprocal rank fusion.
  hybrid_rerank  - take the top 30 hybrid results and let a cross-encoder re-score
                   them (each chunk gets its whole-text or best-paragraph score,
                   whichever is higher), blend that with the hybrid order, keep 5.

Why bother with BM25? Embeddings are great at paraphrases but fuzzy on exact
tokens: model names ("BERT-large"), dataset names ("SQuAD 2.0"), metric names.
Keyword search nails those, and fusion gets us the best of both.

Quick check from the terminal:
    python retrieve.py "Which datasets are used for evaluation?" --mode hybrid
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field

import numpy as np

from config import settings
from ingest import embedding_text

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


class Retriever:
    """Loads the index once and answers retrieval queries in any of the three modes."""

    def __init__(self, collection_name: str | None = None, load_reranker: bool = True):
        import chromadb
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        self.collection_name = collection_name or settings.collection_name()
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        try:
            self.collection = client.get_collection(self.collection_name)
        except Exception as exc:
            raise RuntimeError(
                f"Collection '{self.collection_name}' not found in {settings.chroma_dir}. "
                "Run `python ingest.py` first."
            ) from exc

        # Pull every chunk into memory. With ~1-2k chunks this is a few MB and it
        # lets BM25 and the reranker work without extra round-trips to Chroma.
        dump = self.collection.get(include=["documents", "metadatas"])
        self.ids: list[str] = list(dump["ids"])
        self.chunks: dict[str, dict] = {}
        for cid, doc, meta in zip(dump["ids"], dump["documents"], dump["metadatas"]):
            self.chunks[cid] = {"text": doc, **meta}
        self._paper_of = np.array([self.chunks[c]["paper_id"] for c in self.ids])

        # BM25 indexes the same "title. section. text" string we embedded.
        corpus = [tokenize(embedding_text(self.chunks[c]["title"], self.chunks[c]["section"],
                                          self.chunks[c]["text"])) for c in self.ids]
        self.bm25 = BM25Okapi(corpus)

        # Passages for MaxP reranking, prepared once up front (it's just string splitting).
        self.passages = {c: split_passages(self.chunks[c]["text"]) for c in self.ids}

        self.embedder = SentenceTransformer(settings.embed_model, device=settings.device)
        self._reranker = None
        if load_reranker:
            self._load_reranker()

    # -- helpers -------------------------------------------------------------

    def _load_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(settings.rerank_model, device=settings.device, max_length=512)
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
        """Nearest neighbours by cosine similarity. Returns (chunk_id, similarity)."""
        q = self.embedder.encode(settings.query_instruction + question, normalize_embeddings=True)
        where = {"paper_id": paper_id} if paper_id else None
        if paper_id:
            # Asking Chroma for more results than the filter can return just triggers warnings.
            n = min(n, int((self._paper_of == paper_id).sum()))
        res = self.collection.query(query_embeddings=[q.tolist()], n_results=max(n, 1), where=where,
                                    include=["distances"])
        # Chroma returns cosine *distance*; flip it back to similarity so bigger = better.
        return [(cid, 1.0 - d) for cid, d in zip(res["ids"][0], res["distances"][0])]

    def bm25_search(self, question: str, n: int, paper_id: str | None = None) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(question))
        if paper_id:
            scores = np.where(self._paper_of == paper_id, scores, -np.inf)
        top = np.argsort(-scores)[:n]
        # Drop zero-score hits: they share no words with the question at all.
        return [(self.ids[i], float(scores[i])) for i in top if np.isfinite(scores[i]) and scores[i] > 0]

    def hybrid_search(self, question: str, n: int, paper_id: str | None = None) -> list[tuple[str, float]]:
        pool = settings.fusion_pool
        dense = [cid for cid, _ in self.dense_search(question, pool, paper_id)]
        sparse = [cid for cid, _ in self.bm25_search(question, pool, paper_id)]
        return reciprocal_rank_fusion([dense, sparse], k=settings.rrf_k)[:n]

    def rerank(self, question: str, candidate_ids: list[str], k: int) -> list[tuple[str, float]]:
        """
        Re-score candidates with the cross-encoder, which reads question and text
        together - slower than embeddings, but much sharper.

        Each chunk is scored twice: as a whole, and paragraph by paragraph. How the
        two are combined (settings.rerank_how) and how much the first-stage order
        still counts (settings.rerank_prior) were picked on a separate dev set by
        experiments.py; "max" + prior 0.5 won there.
        """
        if not candidate_ids:
            return []
        model = self._load_reranker()

        # One batched predict() call for every chunk and every passage.
        pairs, owner = [], []
        for c in candidate_ids:
            meta = self.chunks[c]
            pairs.append((question, embedding_text(meta["title"], meta["section"], meta["text"])))
            owner.append((c, "chunk"))
            # A one-paragraph chunk's only passage is the chunk itself - about a quarter of
            # all chunks. No point scoring the same text twice; the chunk score stands in.
            if self.passages[c] == [meta["text"].strip()]:
                continue
            for p in self.passages[c]:
                pairs.append((question, embedding_text(meta["title"], meta["section"], p)))
                owner.append((c, "passage"))
        scores = model.predict(pairs, batch_size=32, show_progress_bar=False)

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
            "maxp": lambda c: best_passage.get(c, chunk_score[c]),
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

    # -- the one function everyone calls -------------------------------------

    def retrieve(self, question: str, mode: str = "hybrid_rerank", k: int | None = None,
                 paper_id: str | None = None) -> list[RetrievedChunk]:
        """Return the top-k chunks for `question` using the chosen mode."""
        k = k or settings.top_k
        if mode == "dense":
            hits, kind = self.dense_search(question, k, paper_id), "dense"
        elif mode == "hybrid":
            hits, kind = self.hybrid_search(question, k, paper_id), "rrf"
        elif mode == "hybrid_rerank":
            candidates = [cid for cid, _ in self.hybrid_search(question, settings.rerank_candidates, paper_id)]
            hits, kind = self.rerank(question, candidates, k), "rerank"
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
