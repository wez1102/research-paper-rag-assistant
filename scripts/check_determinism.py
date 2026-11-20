"""
Run the first-stage retrievers twice, in two separate processes, and compare the
top-50 lists for every question. Round 2 found that Chroma's approximate (HNSW) dense
search gave different lists from one process to the next; dense search is exact now,
and this script is how that was checked.

    python scripts/check_determinism.py --split dev
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def dump(split: str, path: str) -> None:
    sys.path.insert(0, str(ROOT))
    from config import settings
    from evaluate import load_questions
    from retrieve import Retriever
    r = Retriever(collection_name=settings.collection_name(split=split), load_reranker=False)
    out = {q["question_id"]: {"dense": [c for c, _ in r.dense_search(q["question"], 50)],
                              "bm25": [c for c, _ in r.bm25_search(q["question"], 50)]}
           for q in load_questions(split)}
    Path(path).write_text(json.dumps(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--_dump", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args._dump:
        dump(args.split, args._dump)
        return
    with tempfile.TemporaryDirectory() as tmp:
        files = [f"{tmp}/run{i}.json" for i in (1, 2)]
        for f in files:
            subprocess.run([sys.executable, __file__, "--split", args.split, "--_dump", f], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        a, b = (json.loads(Path(f).read_text()) for f in files)
    bad = {k: sum(a[q][k] != b[q][k] for q in a) for k in ("dense", "bm25")}
    print(f"{len(a)} questions; lists that differ between two processes: {bad}")
    sys.exit(1 if any(bad.values()) else 0)


if __name__ == "__main__":
    main()
