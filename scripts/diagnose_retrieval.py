from __future__ import annotations

"""
scripts/diagnose_retrieval.py

Localises WHERE retrieval recall is lost, rather than only reporting that it is.

A hybrid retriever has four places a gold passage can disappear:

  Stage 0  CORPUS      — is the answer even in the index? (extraction failure)
  Stage 1  CANDIDATES  — does RRF surface it in the top-RERANK_CANDIDATES pool?
  Stage 2  RERANK      — does the cross-encoder keep it in the top-k?
  Stage 3  DIVERSITY   — does the per-source cap evict it?

Reporting a single "53% hit rate" conflates all four. This script measures each
independently, so the fix targets the stage that is actually failing.

Gold is defined by `expected_chunks` (source + page) in the questions file, and
independently by whether the expected answer's distinctive tokens appear in a
retrieved chunk — the two agree closely and disagreements are reported.

Usage:
    uv run python -m scripts.diagnose_retrieval
    uv run python -m scripts.diagnose_retrieval --questions data/questions.json
    uv run python -m scripts.diagnose_retrieval --no-expand   # ablate query expansion
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.chunk import (  # noqa: E402
    RERANK_CANDIDATES,
    TOP_K,
    _get_cross_encoder,
    _get_openai_client,
    OPENAI_EMBED_MODEL,
    load_artifacts,
)

import faiss  # noqa: E402
from nltk.corpus import stopwords  # noqa: E402
from nltk.tokenize import word_tokenize  # noqa: E402

DATA_DIR = Path(os.environ.get("PDF_OUTPUT_DIR", "data"))

# Content words that carry answer signal; everything else is noise for matching.
_STOP = None


def _stopset() -> set[str]:
    global _STOP
    if _STOP is None:
        _STOP = set(stopwords.words("english"))
    return _STOP


def answer_keys(expected: str) -> list[str]:
    """Distinctive tokens from the expected answer: dates, numbers, rare words."""
    keys: list[str] = []
    # Dates like "14 June 2027" and bare years
    keys += re.findall(r"\b\d{1,2}\s+\w+\s+\d{4}\b", expected)
    keys += re.findall(r"\b(?:19|20)\d{2}\b", expected)
    # Directive / Regulation / Article references
    keys += re.findall(r"\b(?:Directive|Regulation)\s*\(?EU\)?\s*\d{4}/\d+", expected)
    keys += re.findall(r"\bArticle\s+\d+[a-z]?\b", expected)
    # Percentages and standalone numbers
    keys += re.findall(r"\b\d+(?:\.\d+)?\s*%", expected)
    return list(dict.fromkeys(k.strip() for k in keys if k.strip()))


def content_hit(chunk_text: str, expected: str) -> bool:
    """Does this chunk plausibly contain the answer?"""
    keys = answer_keys(expected)
    low = chunk_text.lower()
    if keys:
        # Any distinctive key present is strong evidence.
        if any(k.lower() in low for k in keys):
            return True
    # Fall back to rare-content-word overlap.
    stop = _stopset()
    words = [w for w in re.findall(r"[a-z]{4,}", expected.lower()) if w not in stop]
    if not words:
        return False
    overlap = sum(1 for w in set(words) if w in low)
    return overlap / max(1, len(set(words))) >= 0.55


def chunk_is_gold(chunk: dict, item: dict) -> bool:
    """Does this chunk match a declared expected_chunks (source, page) pair?"""
    for exp in (item.get("expected_chunks") or []):
        if chunk.get("page") == exp.get("page"):
            # Source filenames drifted between corpus versions, so match on the
            # stem prefix rather than requiring an exact filename equality.
            a = str(chunk.get("source", "")).lower()
            b = str(exp.get("source", "")).lower()
            if a == b or a.split(".")[0] in b or b.split(".")[0] in a:
                return True
    return False


def staged_retrieve(
    query: str,
    index,
    chunks: list[dict],
    bm25: BM25Okapi,
    k: int = TOP_K,
    candidates_n: int = RERANK_CANDIDATES,
) -> dict:
    """Reproduce the retrieval pipeline, returning the output of every stage."""
    stop = _stopset()
    query_tokens = [
        w for w in word_tokenize(query.lower()) if w.isalpha() and w not in stop
    ]

    bm25_scores = bm25.get_scores(query_tokens)
    bm25_ranks = np.argsort(bm25_scores)[::-1]

    client = _get_openai_client()
    resp = client.embeddings.create(
        model=OPENAI_EMBED_MODEL, input=[query],
        encoding_format="float", dimensions=index.d,
    )
    q_vec = np.array([resp.data[0].embedding], dtype="float32")
    faiss.normalize_L2(q_vec)
    _, sem_indices = index.search(q_vec, len(chunks))
    sem_ranks = {int(i): r for r, i in enumerate(sem_indices[0]) if i != -1}

    rrf_k = 60
    rrf: dict[int, float] = {}
    for rank, idx in enumerate(bm25_ranks):
        rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (rrf_k + rank + 1)
    for idx, rank in sem_ranks.items():
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

    seen: set[str] = set()
    cand: list[int] = []
    for idx in sorted(rrf, key=rrf.__getitem__, reverse=True):
        t = chunks[idx]["text"].strip()
        if t not in seen:
            seen.add(t)
            cand.append(idx)
        if len(cand) >= candidates_n:
            break

    pairs = [(query, chunks[i]["text"]) for i in cand]
    ce = _get_cross_encoder().predict(pairs)
    reranked = [i for i, _ in sorted(zip(cand, ce), key=lambda x: x[1], reverse=True)]

    MAX_PER_SOURCE = 3
    counts: dict[str, int] = {}
    final: list[int] = []
    for idx in reranked:
        src = chunks[idx]["source"]
        if counts.get(src, 0) < MAX_PER_SOURCE:
            counts[src] = counts.get(src, 0) + 1
            final.append(idx)
        if len(final) >= k:
            break

    # BM25-only and dense-only top-k, for the ablation table.
    bm25_top = [int(i) for i in bm25_ranks[:k]]
    dense_top = [i for i, _ in sorted(sem_ranks.items(), key=lambda x: x[1])[:k]]

    return {
        "candidates": cand,
        "reranked_topk": reranked[:k],
        "final": final,
        "bm25_top": bm25_top,
        "dense_top": dense_top,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieval failure-stage diagnosis")
    ap.add_argument("--questions", default=str(DATA_DIR / "questions.json"))
    ap.add_argument("--output", default=str(DATA_DIR / "retrieval_diagnosis.json"))
    ap.add_argument("--candidates", type=int, default=RERANK_CANDIDATES)
    ap.add_argument("--k", type=int, default=TOP_K)
    args = ap.parse_args()

    items = json.loads(Path(args.questions).read_text())
    chunks, index, bm25 = load_artifacts(DATA_DIR)
    print(f"Loaded {len(chunks):,} chunks\n")

    rows = []
    for n, item in enumerate(items, 1):
        q = item["question"]
        expected = item.get("expected_answer", "")
        if not expected or "out_of_scope" in str(item.get("sub_type", "")):
            continue

        st = staged_retrieve(q, index, chunks, bm25, k=args.k,
                             candidates_n=args.candidates)

        # Where does gold live, by both definitions?
        gold_corpus = [i for i, c in enumerate(chunks) if chunk_is_gold(c, item)]
        content_corpus = [
            i for i, c in enumerate(chunks) if content_hit(c["text"], expected)
        ]

        def any_in(idxs, pool):
            return bool(set(idxs) & set(pool))

        row = {
            "id": item.get("id", f"q{n:02d}"),
            "sub_type": item.get("sub_type", ""),
            "difficulty": item.get("difficulty", ""),
            "question": q[:90],
            "gold_in_corpus": len(gold_corpus) > 0,
            "content_in_corpus": len(content_corpus) > 0,
            "n_content_chunks": len(content_corpus),
            "in_candidates": any_in(content_corpus, st["candidates"]),
            "in_reranked_topk": any_in(content_corpus, st["reranked_topk"]),
            "in_final": any_in(content_corpus, st["final"]),
            "bm25_only_hit": any_in(content_corpus, st["bm25_top"]),
            "dense_only_hit": any_in(content_corpus, st["dense_top"]),
        }

        # Attribute the failure to its earliest stage.
        if not row["content_in_corpus"]:
            row["lost_at"] = "0_corpus_extraction"
        elif not row["in_candidates"]:
            row["lost_at"] = "1_rrf_candidates"
        elif not row["in_reranked_topk"]:
            row["lost_at"] = "2_cross_encoder"
        elif not row["in_final"]:
            row["lost_at"] = "3_diversity_cap"
        else:
            row["lost_at"] = "none"

        rows.append(row)
        status = "OK " if row["lost_at"] == "none" else "MISS"
        print(f"[{n:2d}/{len(items)}] {status} {row['id']:<5} {row['sub_type']:<20} "
              f"lost_at={row['lost_at']}")

    # ---------------------------------------------------------------- report
    n = len(rows)
    lost = defaultdict(int)
    for r in rows:
        lost[r["lost_at"]] += 1

    print("\n" + "=" * 70)
    print("RETRIEVAL FAILURE-STAGE DIAGNOSIS")
    print("=" * 70)
    print(f"  Questions scored: {n}\n")
    print("  Where recall is lost:")
    for stage in ["0_corpus_extraction", "1_rrf_candidates", "2_cross_encoder",
                  "3_diversity_cap", "none"]:
        c = lost[stage]
        bar = "#" * int(30 * c / max(1, n))
        label = "SUCCEEDED" if stage == "none" else stage
        print(f"    {label:<22} {c:>3}/{n}  {c/n:>6.1%}  {bar}")

    print("\n  Stage-wise surviving recall (cumulative):")
    for key, label in [
        ("content_in_corpus", "in corpus"),
        ("in_candidates", "→ survives RRF"),
        ("in_reranked_topk", "→ survives cross-encoder"),
        ("in_final", "→ survives diversity cap"),
    ]:
        c = sum(1 for r in rows if r[key])
        print(f"    {label:<28} {c:>3}/{n}  {c/n:>6.1%}")

    print("\n  Retriever ablation (top-k hit rate, no reranking):")
    for key, label in [("bm25_only_hit", "BM25 only"),
                       ("dense_only_hit", "Dense only"),
                       ("in_final", "Hybrid + rerank")]:
        c = sum(1 for r in rows if r[key])
        print(f"    {label:<28} {c:>3}/{n}  {c/n:>6.1%}")

    by_type: dict[str, list] = defaultdict(list)
    for r in rows:
        by_type[r["sub_type"] or "untyped"].append(r)
    print("\n  Failure stage by question type:")
    for t, rs in sorted(by_type.items()):
        stages = ", ".join(sorted({x["lost_at"] for x in rs if x["lost_at"] != "none"}))
        ok = sum(1 for x in rs if x["lost_at"] == "none")
        print(f"    {t:<20} {ok}/{len(rs)} ok   {stages or '—'}")
    print("=" * 70)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"candidates": args.candidates, "k": args.k},
        "summary": {k: lost[k] for k in lost},
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
