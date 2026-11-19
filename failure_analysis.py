"""
Why does corpus-wide search (no paper picked) miss so much more than paper search?

Reads the per-question results that evaluate.py wrote for one split and sorts each
corpus-scope miss of the reranked system into one cause, checked in this order:

  unreachable      no chunk "claims" any gold paragraph (chunk boundaries: the
                   evidence is split so thinly that no chunk holds enough of it)
  wrong_paper      none of the top 5 comes from the question's paper
                   (split further by whether the question is generic, e.g.
                   "What baselines do they use?", or names something specific)
  not_in_pool      right paper is there, but the gold chunk isn't among the 30
                   candidates the reranker sees (first-stage recall)
  misranked        the gold chunk was a candidate but ended up below rank 5

It also reports, for every question, how much of the question's vocabulary appears
in its gold evidence (a rough "terminology mismatch" signal) and how many separate
chunks the gold evidence is spread over.

    python failure_analysis.py --split dev --results results/v2/dev/baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import settings
from evaluate import ID_DTYPES, load_questions
from retrieve import tokenize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--results", required=True, help="folder with per_question.csv from evaluate.py")
    ap.add_argument("--mode", default="hybrid_rerank")
    args = ap.parse_args()

    questions = {q["question_id"]: q for q in load_questions(args.split)}
    suffix = "" if args.split == "app" else f"_{args.split}"
    chunks = [json.loads(line) for line in (settings.data_dir / f"chunks_{settings.chunk_size}{suffix}.jsonl").open()]
    labels_path = Path("eval") / f"question_specificity_{args.split}.json"
    labels = json.load(labels_path.open())["labels"] if labels_path.exists() else {}

    # paragraph text per (paper, paragraph id) for the vocabulary-overlap signal
    from ingest import load_papers, paper_paragraphs
    start, n = settings.splits[args.split]
    para_text = {(p["id"], para.idx): para.text for p in load_papers(n, start) for para in paper_paragraphs(p)}

    per = pd.read_csv(Path(args.results) / "per_question.csv", dtype=ID_DTYPES)
    corpus = per[(per["mode"] == args.mode) & (per["scope"] == "corpus")].set_index("question_id")

    # Which chunks claim each gold paragraph?
    claims: dict[tuple[str, int], list[str]] = {}
    for c in chunks:
        for pid in c["paragraph_ids"]:
            claims.setdefault((c["paper_id"], pid), []).append(c["chunk_id"])

    # First-stage candidates (hybrid top 30, corpus scope) for the "not_in_pool" check
    from retrieve import Retriever
    r = Retriever(collection_name=settings.collection_name(split=args.split), load_reranker=False)

    rows = []
    for qid, q in questions.items():
        row = corpus.loc[qid]
        gold = [(q["paper_id"], g) for g in q["gold_paragraph_ids"]]
        gold_chunks = {cid for g in gold for cid in claims.get(g, [])}
        q_terms = set(tokenize(q["question"]))
        gold_terms = set(tokenize(" ".join(para_text.get(g, "") for g in gold)))
        overlap = len(q_terms & gold_terms) / len(q_terms) if q_terms else 0.0
        hit = pd.notna(row["first_hit_rank"])
        cause = "hit"
        if not hit:
            pool = {cid for cid, _ in r.hybrid_search(q["question"], settings.rerank_candidates)}
            if not gold_chunks:
                cause = "unreachable"
            elif pd.isna(row["right_paper_rank"]):
                cause = "wrong_paper"
            elif not (gold_chunks & pool):
                cause = "not_in_pool"
            else:
                cause = "misranked"
        rows.append({"question_id": qid, "question": q["question"], "specificity": labels.get(qid, "?"),
                     "cause": cause, "gold_chunks": len(gold_chunks),
                     "gold_paragraphs": len(q["gold_paragraph_ids"]), "q_term_overlap": round(overlap, 3)})
    df = pd.DataFrame(rows)
    out = Path(args.results) / "corpus_failure_analysis.csv"
    df.to_csv(out, index=False)

    print(f"{args.split}: {len(df)} questions, corpus-scope {args.mode}")
    print(pd.crosstab(df["cause"], df["specificity"], margins=True).to_string())
    print("\nHit@5 by question type:")
    for spec, g in df.groupby("specificity"):
        print(f"  {spec:<9} n={len(g):>3}  hit@5={(g['cause'] == 'hit').mean():.3f}")
    print("\nVocabulary overlap with gold evidence (share of question terms found in it):")
    for cause, g in df.groupby("cause"):
        print(f"  {cause:<12} n={len(g):>3}  median overlap={g['q_term_overlap'].median():.2f}  "
              f"gold spread over >1 chunk: {(g['gold_chunks'] > 1).mean():.0%}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
