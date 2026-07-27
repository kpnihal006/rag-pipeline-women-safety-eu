from __future__ import annotations

"""
scripts/experiment_grid.py

Factorial comparison of corpus-processing and retrieval design decisions.

Everything here runs locally and free: embeddings come from sentence-transformers
models, retrieval is FAISS + BM25 in-process, and no hosted API is called.

FACTORS
    embedding model   all-MiniLM-L6-v2 | bge-small-en-v1.5 | e5-small-v2
    chunk size        400 | 800 | 1200 characters
    chunk overlap     0 | 100 | 200 characters
    structure         whether metadata headers (dates, directives, articles)
                      are prepended to the embedded text
    retriever         dense | bm25 | hybrid (RRF)
    k                 3 | 5 | 8 | 12

MEASUREMENT
Gold is defined at **page** level: a configuration scores a hit when it returns
any chunk from a page the ground truth names. Page-level gold is what makes
chunk-size comparison valid at all — chunk indices change when you re-chunk, so
a chunk-id-based gold would silently redefine the target between conditions.

Metrics: recall@k, plus MRR and precision@k for the ranking quality that recall
alone hides.

Because each factor combination requires re-chunking and re-embedding the
corpus, the grid runs over a document subset: every document referenced by the
ground truth, plus a fixed random sample of others as distractors. The subset
is held constant across all conditions, so comparisons are like-for-like even
though absolute numbers differ from the full corpus.

Usage:
    uv run python -m scripts.experiment_grid --stage chunking
    uv run python -m scripts.experiment_grid --stage k
    uv run python -m scripts.experiment_grid --stage all --distractors 12
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from rank_bm25 import BM25Okapi

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.metadata import extract_metadata, metadata_header  # noqa: E402

DATA_DIR = Path(os.environ.get("PDF_OUTPUT_DIR", "data"))

EMBEDDERS = {
    "MiniLM-L6": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "e5-small": "intfloat/e5-small-v2",
}

_model_cache: dict[str, object] = {}
_stop: set[str] | None = None


def stopset() -> set[str]:
    global _stop
    if _stop is None:
        _stop = set(stopwords.words("english"))
    return _stop


def get_model(name: str):
    if name not in _model_cache:
        from sentence_transformers import SentenceTransformer

        print(f"    loading {EMBEDDERS[name]} …", flush=True)
        _model_cache[name] = SentenceTransformer(EMBEDDERS[name])
    return _model_cache[name]


def encode(name: str, texts: list[str], is_query: bool = False) -> np.ndarray:
    """Encode with the model's required prefix convention.

    e5 and bge are trained with asymmetric query/passage prefixes; omitting
    them measures the model badly rather than measuring a bad model.
    """
    model = get_model(name)
    if name == "e5-small":
        texts = [("query: " if is_query else "passage: ") + t for t in texts]
    elif name == "bge-small" and is_query:
        texts = ["Represent this sentence for searching relevant passages: " + t
                 for t in texts]
    v = model.encode(texts, batch_size=64, show_progress_bar=False,
                     convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(v)
    return v


# ---------------------------------------------------------------------------
# Chunking under test
# ---------------------------------------------------------------------------

def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Sentence-aware chunker — the production strategy, parameterised."""
    if not text or not text.strip():
        return []
    sentences = sent_tokenize(text)
    chunks, current, cur_len = [], [], 0
    for sent in sentences:
        if cur_len + len(sent) > size and current:
            chunks.append(" ".join(current))
            tail, tail_len = [], 0
            for s in reversed(current):
                if tail_len + len(s) <= overlap:
                    tail.insert(0, s)
                    tail_len += len(s)
                else:
                    break
            current, cur_len = tail, tail_len
        current.append(sent)
        cur_len += len(sent)
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_chunks(pages: list[dict], size: int, overlap: int,
                 structure: bool) -> list[dict]:
    out: list[dict] = []
    for entry in pages:
        for i, text in enumerate(split_text(entry["text"], size, overlap)):
            rec = {"source": entry["source"], "page": entry["page"],
                   "chunk_index": i, "text": text, "embed_text": text}
            if structure:
                header = metadata_header(extract_metadata(text))
                if header:
                    rec["embed_text"] = f"{header}\n\n{text}"
            out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Retrieval under test
# ---------------------------------------------------------------------------

def rank_dense(qv: np.ndarray, index, n: int) -> list[int]:
    _, idx = index.search(qv, min(n, index.ntotal))
    return [int(i) for i in idx[0] if i != -1]


def rank_bm25(query: str, bm25: BM25Okapi, n: int) -> list[int]:
    toks = [w for w in word_tokenize(query.lower())
            if w.isalpha() and w not in stopset()]
    scores = bm25.get_scores(toks)
    return [int(i) for i in np.argsort(scores)[::-1][:n]]


def rrf(a: list[int], b: list[int], k: int = 60) -> list[int]:
    s: dict[int, float] = {}
    for r, i in enumerate(a):
        s[i] = s.get(i, 0.0) + 1.0 / (k + r + 1)
    for r, i in enumerate(b):
        s[i] = s.get(i, 0.0) + 1.0 / (k + r + 1)
    return sorted(s, key=s.__getitem__, reverse=True)


# ---------------------------------------------------------------------------
# Scoring — page-level gold
# ---------------------------------------------------------------------------

def gold_pages(item: dict) -> set[tuple[str, int]]:
    out = set()
    for e in (item.get("expected_chunks") or []):
        src, page = e.get("source"), e.get("page")
        if src and page is not None:
            out.add((str(src).lower(), int(page)))
    return out


def score(ranked: list[int], chunks: list[dict], gold: set, k: int) -> dict:
    topk = ranked[:k]
    hit, rr, n_rel = False, 0.0, 0
    for rank, idx in enumerate(topk, 1):
        c = chunks[idx]
        key = (str(c["source"]).lower(), int(c["page"]))
        rel = any(key[1] == g[1] and (key[0] == g[0] or
                  key[0].split(".")[0] in g[0] or g[0].split(".")[0] in key[0])
                  for g in gold)
        if rel:
            n_rel += 1
            if not hit:
                hit, rr = True, 1.0 / rank
    return {"hit": hit, "rr": rr, "precision": n_rel / max(1, k)}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def load_questions(paths: list[Path]) -> list[dict]:
    items: list[dict] = []
    for p in paths:
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for it in data:
            if gold_pages(it) and it.get("question"):
                it["_set"] = p.stem
                items.append(it)
    return items


def build_subset(corpus: list[dict], items: list[dict], distractors: int,
                 seed: int = 20260727) -> list[dict]:
    """Every ground-truth document, plus a fixed sample of others."""
    needed = {g[0] for it in items for g in gold_pages(it)}

    by_src: dict[str, list[dict]] = defaultdict(list)
    for e in corpus:
        by_src[e["source"]].append(e)

    def matches(src: str) -> bool:
        s = src.lower()
        return any(s == n or s.split(".")[0] in n or n.split(".")[0] in s
                   for n in needed)

    keep = [s for s in by_src if matches(s)]
    others = sorted(s for s in by_src if s not in keep)
    random.Random(seed).shuffle(others)
    keep += others[:distractors]

    pages = [e for s in keep for e in by_src[s]]
    return pages


def main() -> None:
    ap = argparse.ArgumentParser(description="Design-decision factorial")
    ap.add_argument("--stage", default="all",
                    choices=["all", "chunking", "embedding", "k", "structure"])
    ap.add_argument("--distractors", type=int, default=12)
    ap.add_argument("--output", default=str(DATA_DIR / "experiment_grid.json"))
    args = ap.parse_args()

    corpus = json.loads((DATA_DIR / "corpus.json").read_text(encoding="utf-8"))
    items = load_questions([DATA_DIR / "questions.json",
                            DATA_DIR / "ground_truth_silver.json"])
    if not items:
        sys.exit("No ground-truth questions with page-level gold found.")

    pages = build_subset(corpus, items, args.distractors)
    docs = len({p["source"] for p in pages})
    print(f"Ground truth : {len(items)} questions "
          f"({', '.join(sorted({i['_set'] for i in items}))})")
    print(f"Subset       : {len(pages):,} pages across {docs} documents\n")

    # ---- factor levels per stage
    if args.stage == "chunking":
        grid = [(e, s, o, st) for e in ["MiniLM-L6"]
                for s in [400, 800, 1200] for o in [0, 100, 200] for st in [False]]
        ks = [8]
    elif args.stage == "embedding":
        grid = [(e, 800, 200, False) for e in EMBEDDERS]
        ks = [8]
    elif args.stage == "structure":
        grid = [(e, s, 200, st) for e in ["MiniLM-L6"]
                for s in [400, 800] for st in [False, True]]
        ks = [8]
    elif args.stage == "k":
        grid = [("MiniLM-L6", 800, 200, False)]
        ks = [3, 5, 8, 12, 20]
    else:
        grid = [(e, s, o, st) for e in EMBEDDERS
                for s in [400, 800, 1200] for o in [100, 200] for st in [False, True]]
        ks = [3, 5, 8, 12]

    print(f"Running {len(grid)} index build(s) × {len(ks)} k value(s) "
          f"× 3 retrievers\n")

    results = []
    for n, (emb, size, overlap, structure) in enumerate(grid, 1):
        t0 = time.perf_counter()
        chunks = build_chunks(pages, size, overlap, structure)
        vecs = encode(emb, [c["embed_text"] for c in chunks])
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)

        tokenized = [[w for w in word_tokenize(c["text"].lower())
                      if w.isalpha() and w not in stopset()] for c in chunks]
        bm25 = BM25Okapi(tokenized)

        qvecs = encode(emb, [it["question"] for it in items], is_query=True)
        build_s = time.perf_counter() - t0

        for retriever in ["dense", "bm25", "hybrid"]:
            per_k: dict[int, list[dict]] = {k: [] for k in ks}
            for qi, it in enumerate(items):
                gold = gold_pages(it)
                maxk = max(ks)
                if retriever == "dense":
                    ranked = rank_dense(qvecs[qi:qi + 1], index, maxk)
                elif retriever == "bm25":
                    ranked = rank_bm25(it["question"], bm25, maxk)
                else:
                    ranked = rrf(rank_dense(qvecs[qi:qi + 1], index, 200),
                                 rank_bm25(it["question"], bm25, 200))[:maxk]
                for k in ks:
                    per_k[k].append(score(ranked, chunks, gold, k))

            for k in ks:
                rows = per_k[k]
                by_set: dict[str, list] = defaultdict(list)
                for it, r in zip(items, rows):
                    by_set[it["_set"]].append(r["hit"])
                results.append({
                    "embedder": emb, "dim": int(vecs.shape[1]),
                    "chunk_size": size, "overlap": overlap,
                    "structure": structure, "retriever": retriever, "k": k,
                    "n_chunks": len(chunks),
                    "recall": round(sum(r["hit"] for r in rows) / len(rows), 4),
                    "mrr": round(sum(r["rr"] for r in rows) / len(rows), 4),
                    "precision": round(
                        sum(r["precision"] for r in rows) / len(rows), 4),
                    "by_set": {s: round(sum(v) / len(v), 3)
                               for s, v in by_set.items()},
                    "build_s": round(build_s, 1),
                })

        best = max(r["recall"] for r in results[-len(ks) * 3:])
        print(f"[{n:2d}/{len(grid)}] {emb:<10} size={size:<5} ov={overlap:<4} "
              f"struct={str(structure):<5} chunks={len(chunks):<6} "
              f"best recall={best:.1%}  ({build_s:.0f}s)", flush=True)

    Path(args.output).write_text(
        json.dumps({"n_questions": len(items), "n_pages": len(pages),
                    "results": results}, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- report
    def table(title: str, keyfn, rows, hold: dict | None = None):
        """Marginalise over one factor, holding confounders fixed.

        Holding matters: BM25 rows are identical across embedding models
        because BM25 never touches an embedding. Taking a max over all
        retrievers would therefore report the same BM25 number for every
        embedder and present it as an embedding comparison.
        """
        if hold:
            rows = [r for r in rows if all(r.get(k) == v for k, v in hold.items())]
        if not rows:
            print(f"\n  {title}: no rows after holding {hold}")
            return
        agg: dict = defaultdict(list)
        for r in rows:
            agg[keyfn(r)].append(r)
        if hold:
            title += "   [held: " + ", ".join(f"{k}={v}" for k, v in hold.items()) + "]"
        print(f"\n  {title}")
        print(f"    {'level':<26} {'recall':>8} {'mrr':>7} {'prec':>7}")
        print("    " + "-" * 50)
        for level in sorted(agg, key=lambda x: -max(r["recall"] for r in agg[x])):
            rs = agg[level]
            b = max(rs, key=lambda r: r["recall"])
            print(f"    {str(level):<26} {b['recall']:>7.1%} "
                  f"{b['mrr']:>7.3f} {b['precision']:>7.3f}")

    print("\n" + "=" * 72)
    print("DESIGN-DECISION COMPARISON  (best cell per level)")
    print("=" * 72)
    if len({r["embedder"] for r in results}) > 1:
        # Dense only — BM25 is embedding-independent and would flatten this.
        table("Embedding model", lambda r: f"{r['embedder']} ({r['dim']}d)",
              results, hold={"retriever": "dense"})
    if len({r["chunk_size"] for r in results}) > 1:
        table("Chunk size", lambda r: f"{r['chunk_size']} chars",
              results, hold={"retriever": "dense"})
        table("Chunk size", lambda r: f"{r['chunk_size']} chars",
              results, hold={"retriever": "hybrid"})
    if len({r["overlap"] for r in results}) > 1:
        table("Chunk overlap", lambda r: f"{r['overlap']} chars",
              results, hold={"retriever": "dense"})
    if len({r["structure"] for r in results}) > 1:
        table("Structure preservation", lambda r: f"metadata={r['structure']}",
              results, hold={"retriever": "dense"})
    if len({r["k"] for r in results}) > 1:
        table("k (passages retrieved)", lambda r: f"k={r['k']}", results)
    table("Retriever", lambda r: r["retriever"], results)

    top = sorted(results, key=lambda r: -r["recall"])[:8]
    print("\n  Top configurations")
    print(f"    {'emb':<11}{'size':>5}{'ov':>5}{'str':>6}{'retr':>8}{'k':>4}"
          f"{'recall':>9}{'mrr':>8}")
    print("    " + "-" * 56)
    for r in top:
        print(f"    {r['embedder']:<11}{r['chunk_size']:>5}{r['overlap']:>5}"
              f"{str(r['structure']):>6}{r['retriever']:>8}{r['k']:>4}"
              f"{r['recall']:>8.1%}{r['mrr']:>8.3f}")
    print("=" * 72)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
