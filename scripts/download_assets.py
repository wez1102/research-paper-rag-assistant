"""
Download the dataset file and both models into one folder, so ingestion and the
server can run with no network (HF_HUB_OFFLINE=1). Used by the Dockerfile and CI;
handy on a laptop before a flight too.

    python scripts/download_assets.py assets/
    export QASPER_PARQUET=assets/qasper/validation.parquet \\
           EMBED_MODEL=assets/models/bge-small-en-v1.5 \\
           RERANK_MODEL=assets/models/ms-marco-MiniLM-L-6-v2

Set EMBED_MODEL_REVISION / RERANK_MODEL_REVISION to a commit hash to pin the exact
model files (docs/REPRODUCIBILITY.md lists the SHA-256 of the files we evaluated).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

MODELS = {
    "bge-small-en-v1.5": ("BAAI/bge-small-en-v1.5", os.getenv("EMBED_MODEL_REVISION")),
    "ms-marco-MiniLM-L-6-v2": ("cross-encoder/ms-marco-MiniLM-L-6-v2", os.getenv("RERANK_MODEL_REVISION")),
}


def main(dest: str) -> None:
    root = Path(dest)
    (root / "qasper").mkdir(parents=True, exist_ok=True)
    src = hf_hub_download("allenai/qasper", "qasper/validation/0000.parquet", repo_type="dataset",
                          revision="refs/convert/parquet")
    shutil.copy(src, root / "qasper" / "validation.parquet")
    for folder, (repo, rev) in MODELS.items():
        # Weights and tokenizer only; skip the ONNX/OpenVINO/TF copies some repos ship.
        snapshot_download(repo, revision=rev, local_dir=root / "models" / folder,
                          allow_patterns=["*.json", "*.txt", "*.safetensors", "1_Pooling/*"])
    print(f"assets ready in {root.resolve()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "assets")
