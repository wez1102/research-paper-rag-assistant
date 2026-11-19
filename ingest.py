"""
Step 1 of the pipeline: turn QASPER papers into an index we can search.

What this script does, in order:
  1. Downloads the QASPER validation split (Parquet) and keeps the first N papers.
  2. Flattens every paper into paragraphs, remembering which section each one came from.
  3. Cuts each section into ~450-token chunks with ~50 tokens of overlap.
     Chunks never cross a section boundary, so they can't cross papers either.
  4. Embeds the chunks with bge-small and stores them in a persistent ChromaDB
     collection, together with paper id / title / section / chunk id metadata.
  5. Writes the evaluation questions (with their gold evidence paragraphs) to
     data/questions.jsonl so evaluate.py can score retrieval automatically.

Run it once:   python ingest.py                 (the 50 papers the app serves)
               python ingest.py --split dev     (tuning papers, for evaluate.py)
               python ingest.py --split final   (held-out papers, for evaluate.py)

Re-running is safe. Each collection carries a manifest (a hash of every chunk id,
chunk text and the embedding model). If nothing changed, the build is skipped;
otherwise a fresh collection is built next to the old one and only swapped in once
it is complete, so a crash halfway never leaves a half-filled or duplicated index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from config import settings

log = logging.getLogger("ingest")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Paragraph:
    paper_id: str
    title: str
    section: str
    idx: int          # position of the paragraph inside its paper (0 = abstract)
    text: str


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    title: str
    section: str
    text: str
    paragraph_ids: list[int]   # which paragraphs (fully or partly) ended up in this chunk
    n_tokens: int


def normalize(text: str) -> str:
    """Collapse whitespace. QASPER evidence strings and paragraphs differ only in spacing."""
    return " ".join(text.split())


def clean_section(name: str | None) -> str:
    # QASPER encodes nested sections as "Method ::: Encoder". A ">" reads nicer in citations.
    if not name:
        return "Untitled section"
    return name.replace(" ::: ", " > ").strip()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_papers(num_papers: int, start: int = 0) -> list[dict]:
    """
    Return `num_papers` rows of the QASPER validation split as plain dicts, starting
    at row `start`. The app uses rows 0-49; experiments.py uses 50-99 as a separate
    dev set so we can tune settings without peeking at the test questions.
    """
    if settings.qasper_local_path:
        path = Path(settings.qasper_local_path)
        log.info("Reading QASPER from local file %s", path)
    else:
        # hf_hub_download caches the file, so later runs don't download it again.
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(
            repo_id=settings.qasper_repo,
            repo_type="dataset",
            revision=settings.qasper_revision,
            filename=settings.qasper_file,
        ))
    df = pd.read_parquet(path)
    # Taking the first N (rather than a random sample) keeps the corpus identical on every machine.
    return df.iloc[start:start + num_papers].to_dict(orient="records")


def paper_paragraphs(paper: dict) -> list[Paragraph]:
    """Flatten one paper into an ordered list of paragraphs (abstract first)."""
    pid, title = paper["id"], paper["title"]
    paras: list[Paragraph] = []

    # The abstract isn't part of full_text in QASPER, but questions do sometimes
    # point at it, so we index it as its own little section.
    if paper.get("abstract"):
        paras.append(Paragraph(pid, title, "Abstract", 0, normalize(paper["abstract"])))

    full = paper["full_text"]
    for section, section_paras in zip(full["section_name"], full["paragraphs"]):
        for p in section_paras:
            if p is None or not p.strip():
                continue
            paras.append(Paragraph(pid, title, clean_section(section), len(paras), normalize(p)))
    return paras


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_section(paras: list[Paragraph], tokenizer, size: int, overlap: int) -> list[tuple[str, list[int], int]]:
    """
    Split the paragraphs of ONE section into overlapping token windows.

    We tokenize with offsets so we can map tokens back to the original characters.
    That matters because bge's tokenizer lowercases everything - decoding token ids
    would give us lowercase text with broken spacing, which looks awful in the UI.

    Returns a list of (chunk_text, paragraph_ids, n_tokens).
    """
    # One entry per token: (paragraph index in `paras`, char start, char end)
    tokens: list[tuple[int, int, int]] = []
    boundaries: set[int] = set()   # token positions where a new paragraph starts
    para_len: list[int] = []       # token count of each paragraph
    for i, p in enumerate(paras):
        boundaries.add(len(tokens))
        enc = tokenizer(p.text, add_special_tokens=False, return_offsets_mapping=True)
        tokens.extend((i, s, e) for s, e in enc["offset_mapping"])
        para_len.append(len(enc["offset_mapping"]))
    n = len(tokens)
    if n == 0:
        return []

    out = []
    start = 0
    while start < n:
        end = min(start + size, n)
        if end < n:
            # Prefer to end on a paragraph boundary as long as the chunk stays at
            # least half full; otherwise just cut mid-paragraph.
            candidates = [b for b in boundaries if start + size // 2 <= b <= end]
            if candidates:
                end = max(candidates)

        window = tokens[start:end]
        # Rebuild the text paragraph by paragraph from the character offsets.
        pieces, para_ids = [], []
        for pi in dict.fromkeys(t[0] for t in window):     # ordered unique
            spans = [t for t in window if t[0] == pi]
            text = paras[pi].text[spans[0][1]:spans[-1][2]]
            pieces.append(text)
            # Only "claim" a paragraph if the chunk holds a real part of it. Otherwise
            # the 50-token overlap tail would let two chunks both count as containing
            # the same evidence, which inflates the hit-rate numbers.
            if len(spans) >= 0.5 * para_len[pi] or len(spans) >= 100:
                para_ids.append(paras[pi].idx)
        out.append(("\n\n".join(pieces), para_ids, len(window)))

        if end >= n:
            break
        # Step back by `overlap` tokens so context isn't lost at the edges,
        # but always move forward by at least one token to avoid an endless loop.
        start = max(end - overlap, start + 1)
    return out


def chunk_paper(paras: list[Paragraph], tokenizer, size: int, overlap: int) -> list[Chunk]:
    """Chunk a whole paper section by section."""
    chunks: list[Chunk] = []
    # Group consecutive paragraphs that share a section name.
    groups: list[list[Paragraph]] = []
    for p in paras:
        if groups and groups[-1][0].section == p.section:
            groups[-1].append(p)
        else:
            groups.append([p])

    for group in groups:
        for text, para_ids, n_tok in chunk_section(group, tokenizer, size, overlap):
            first = group[0]
            chunks.append(Chunk(
                chunk_id=f"{first.paper_id}-{len(chunks):03d}",
                paper_id=first.paper_id,
                title=first.title,
                section=first.section,
                text=text,
                paragraph_ids=para_ids,
                n_tokens=n_tok,
            ))
    return chunks


def embedding_text(title: str, section: str, text: str) -> str:
    """
    What actually gets embedded / keyword-indexed. Prepending the title and section
    gives each chunk some context ("which paper am I from?") - a cheap trick that
    helps a lot when questions mention the paper's topic.
    """
    return f"{title}. {section}.\n{text}"


# ---------------------------------------------------------------------------
# Evaluation questions
# ---------------------------------------------------------------------------

def reference_answers(answer_list: list[dict]) -> list[str]:
    """Turn QASPER's structured answers into plain reference strings (one per annotator)."""
    refs = []
    for a in answer_list:
        if a["unanswerable"]:
            continue
        if a["extractive_spans"] is not None and len(a["extractive_spans"]) > 0:
            refs.append(", ".join(a["extractive_spans"]))
        elif a["free_form_answer"]:
            refs.append(a["free_form_answer"])
        elif a["yes_no"] is not None:
            refs.append("Yes" if a["yes_no"] else "No")
    return refs


def build_questions(paper: dict, paras: list[Paragraph],
                    unanswerable: list[dict] | None = None) -> tuple[list[dict], dict]:
    """
    Pull the answerable questions for one paper and map their evidence strings to
    paragraph indices. Questions whose only evidence is a figure/table
    ("FLOAT SELECTED: ...") are skipped, since we don't index figures.

    If `unanswerable` is a list, questions that every annotator marked unanswerable
    are appended to it. They're useful too: the right behaviour there is to refuse.
    """
    by_text = {p.text: p.idx for p in paras}
    stats = {"total": 0, "unanswerable": 0, "float_only": 0, "unmapped": 0, "kept": 0}
    questions = []

    qas = paper["qas"]
    for qid, question, answers in zip(qas["question_id"], qas["question"], qas["answers"]):
        stats["total"] += 1
        answer_list = list(answers["answer"])
        answerable = [a for a in answer_list if not a["unanswerable"]]
        if not answerable:
            stats["unanswerable"] += 1
            if unanswerable is not None:
                unanswerable.append({"question_id": qid, "paper_id": paper["id"], "title": paper["title"],
                                     "question": question})
            continue

        evidence = [normalize(e) for a in answerable for e in a["evidence"]
                    if e and not e.startswith("FLOAT SELECTED")]
        if not evidence:
            stats["float_only"] += 1
            continue

        gold: set[int] = set()
        for ev in evidence:
            if ev in by_text:
                gold.add(by_text[ev])
            else:
                # A handful of evidence strings are trimmed or merged versions of a
                # paragraph, so fall back to substring containment either way round.
                gold.update(p.idx for p in paras if ev in p.text or (len(p.text) > 40 and p.text in ev))
        if not gold:
            stats["unmapped"] += 1
            continue

        stats["kept"] += 1
        questions.append({
            "question_id": qid,
            "paper_id": paper["id"],
            "title": paper["title"],
            "question": question,
            "gold_paragraph_ids": sorted(gold),
            "references": reference_answers(answer_list),
        })
    return questions, stats


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def model_tag(model: str) -> str:
    """'BAAI/bge-small-en-v1.5' and '/models/bge-small-en-v1.5' are the same model to us."""
    return model.rstrip("/").split("/")[-1]


def index_manifest(chunks: list[Chunk], chunk_size: int, chunk_overlap: int) -> dict:
    """
    Everything that decides what's in the index, boiled down to one hash. Two builds
    with the same manifest hash hold exactly the same records.
    """
    h = hashlib.sha256()
    h.update(f"{model_tag(settings.embed_model)}|{chunk_size}|{chunk_overlap}".encode())
    for c in chunks:
        h.update(b"\x00" + c.chunk_id.encode() + b"\x00" + embedding_text(c.title, c.section, c.text).encode())
    return {
        "manifest_hash": h.hexdigest(),
        "embed_model": model_tag(settings.embed_model),
        "n_chunks": len(chunks),
        "n_papers": len({c.paper_id for c in chunks}),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


def index_is_current(client, collection_name: str, manifest: dict) -> bool:
    """True if the collection already holds exactly what we're about to build."""
    try:
        col = client.get_collection(collection_name)
    except Exception:
        return False
    meta = col.metadata or {}
    return meta.get("manifest_hash") == manifest["manifest_hash"] and col.count() == manifest["n_chunks"]


def build_index(chunks: list[Chunk], collection_name: str, embedder, manifest: dict,
                force: bool = False) -> bool:
    """
    Embed chunks and (re)create the Chroma collection. Returns False if the existing
    collection was already up to date and nothing had to be done.

    The new records go into a temporary collection first. Only when it's complete
    (and the count checks out) is the old one dropped and the new one renamed into
    place. So a crash or Ctrl-C mid-build leaves the previous index untouched, and
    a re-run can never add a second copy of the same chunks.
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    if not force and index_is_current(client, collection_name, manifest):
        log.info("Collection '%s' is already up to date (manifest %s...) - nothing to do.",
                 collection_name, manifest["manifest_hash"][:12])
        return False

    ids = [c.chunk_id for c in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate chunk ids - refusing to build an index with duplicate records.")

    tmp_name = f"{collection_name}__building"
    try:
        client.delete_collection(tmp_name)          # leftovers from a crashed run
    except Exception:
        pass
    collection = client.create_collection(tmp_name, metadata={"hnsw:space": "cosine", **manifest})

    texts = [embedding_text(c.title, c.section, c.text) for c in chunks]
    t0 = time.perf_counter()
    vectors = embedder.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    log.info("Embedded %d chunks in %.1fs", len(chunks), time.perf_counter() - t0)

    # Chroma metadata values have to be scalars, so the paragraph list becomes a string.
    metadatas = [{
        "paper_id": c.paper_id,
        "title": c.title,
        "section": c.section,
        "chunk_id": c.chunk_id,
        "paragraph_ids": ",".join(map(str, c.paragraph_ids)),
        "n_tokens": c.n_tokens,
    } for c in chunks]

    batch = 1000
    for i in range(0, len(chunks), batch):
        # upsert rather than add: even if a batch were somehow sent twice, an id can
        # only ever exist once.
        collection.upsert(
            ids=ids[i:i + batch],
            documents=[c.text for c in chunks[i:i + batch]],
            embeddings=vectors[i:i + batch].tolist(),
            metadatas=metadatas[i:i + batch],
        )
    if collection.count() != len(chunks):
        raise RuntimeError(f"Built {collection.count()} records but expected {len(chunks)}; "
                           f"the old index '{collection_name}' was left as it was.")

    # Swap: drop the old collection, give the new one its real name.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection.modify(name=collection_name)
    log.info("Collection '%s' now holds %d chunks (manifest %s...)", collection_name,
             collection.count(), manifest["manifest_hash"][:12])
    return True


def build_split(split: str, tokenizer, chunk_size: int, chunk_overlap: int, num_papers: int | None = None):
    """Chunks, answerable questions, unanswerable questions and stats for one paper slice."""
    start, n = settings.splits[split]
    papers = load_papers(num_papers or n, start)
    log.info("Split '%s': QASPER validation rows %d-%d (%d papers)", split, start, start + len(papers) - 1,
             len(papers))
    chunks: list[Chunk] = []
    questions: list[dict] = []
    unanswerable: list[dict] = []
    totals: dict[str, int] = {}
    for paper in papers:
        paras = paper_paragraphs(paper)
        chunks.extend(chunk_paper(paras, tokenizer, chunk_size, chunk_overlap))
        qs, stats = build_questions(paper, paras, unanswerable)
        questions.extend(qs)
        for k, v in stats.items():
            totals[k] = totals.get(k, 0) + v
    return chunks, questions, unanswerable, totals


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build the QASPER chunk index.")
    parser.add_argument("--split", choices=list(settings.splits), default="app",
                        help="Which paper slice to index (app = what the web app serves).")
    parser.add_argument("--num-papers", type=int, default=None,
                        help="Override the number of papers in the slice (default: the whole slice).")
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--force", action="store_true", help="Rebuild even if the index is up to date.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    # Imported here so `import ingest` in tests stays fast.
    from sentence_transformers import SentenceTransformer
    from transformers.utils import logging as hf_logging

    # We tokenize whole paragraphs just to count tokens, which makes transformers
    # warn about sequences longer than 512. Harmless here, so keep the log clean.
    hf_logging.set_verbosity_error()

    embedder = SentenceTransformer(settings.embed_model, device=settings.device)
    num_papers = args.num_papers or (settings.num_papers if args.split == "app" else None)
    chunks, questions, unanswerable, totals = build_split(args.split, embedder.tokenizer, args.chunk_size,
                                                          args.chunk_overlap, num_papers)

    tok = [c.n_tokens for c in chunks]
    log.info("Made %d chunks (avg %.0f tokens, max %d)", len(chunks), sum(tok) / len(tok), max(tok))
    log.info("Questions: %s", totals)

    manifest = index_manifest(chunks, args.chunk_size, args.chunk_overlap)
    build_index(chunks, settings.collection_name(args.chunk_size, args.split), embedder, manifest, args.force)

    # Questions only depend on the papers, not the chunk size, so one file serves every experiment.
    q_path = settings.questions_path(args.split)
    with q_path.open("w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")
    with settings.unanswerable_path(args.split).open("w") as f:
        for q in unanswerable:
            f.write(json.dumps(q) + "\n")
    log.info("Wrote %d evaluation questions to %s (+%d unanswerable)", len(questions), q_path, len(unanswerable))

    # A plain JSONL copy of the chunks is useful for eyeballing what the index contains.
    suffix = "" if args.split == "app" else f"_{args.split}"
    with (settings.data_dir / f"chunks_{args.chunk_size}{suffix}.jsonl").open("w") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")


if __name__ == "__main__":
    main()
