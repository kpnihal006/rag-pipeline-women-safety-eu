from __future__ import annotations

"""
scripts/experiments.py

Retrieval ablation harness.

The failure-stage diagnosis (scripts/diagnose_retrieval.py) showed that recall
is lost at the re-ranking stage, not at extraction: RRF surfaces the gold
passage for 83% of questions, and the cross-encoder then demotes it out of the
top-k, ending at 61%. Dense-only retrieval without any re-ranking scored higher
than the full pipeline.

That is a configuration problem, so this script measures configurations rather
than arguing about them. It sweeps:

  retriever    bm25 | dense | hybrid (RRF)
  reranker     none | a cross-encoder model | blended (CE + RRF prior)
  candidates   size of the pool handed to the re-ranker
  k            final result count

and reports recall@k for each. Query embeddings and cross-encoder models are
cached across configurations, so a full sweep costs one embedding call per
question in total.

Recall is measured with the same content-matching function used by the
diagnosis script, so the numbers are directly comparable.

Usage:
    uv run python -m scripts.experiments                    # default sweep
    uv run python -m scripts.experiments --quick            # small sweep
    uv run python -m scripts.experiments --rerankers ms-marco-MiniLM-L-6-v2
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.chunk import (  # noqa: E402
    OPENAI_EMBED_MODEL,
    _get_openai_client,
    load_artifacts,
)
from scripts.diagnose_retrieval import content_hit  # noqa: E402

DATA_DIR = Path(os.environ.get("PDF_OUTPUT_DIR", "data"))

#: Re-ranker candidates. L-6 is the incumbent; the others are stronger models
#: with the same interface, so swapping is a one-line change if one wins.
RERANKER_MODELS = {
    "minilm-L6": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "minilm-L12": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "bge-base": "BAAI/bge-reranker-base",
}

_ce_cache: dict[str, object] = {}
#: (reranker, qid, pool-fingerprint) -> scores. The blend and k sweeps reuse
#: one cross-encoder pass instead of repeating it for every variant.
_score_cache: dict[tuple, np.ndarray] = {}
_embed_cache: dict[str, np.ndarray] = {}
_stop: set[str] | None = None


def stopset() -> set[str]:
    global _stop
    if _stop is None:
        _stop = set(stopwords.words("english"))
    return _stop


def get_ce(name: str):
    """Load and cache a cross-encoder by short name."""
    if name not in _ce_cache:
        from sentence_transformers import CrossEncoder

        print(f"    loading reranker {RERANKER_MODELS[name]} …", flush=True)
        _ce_cache[name] = CrossEncoder(RERANKER_MODELS[name])
    return _ce_cache[name]


def embed(query: str, dim: int) -> np.ndarray:
    key = f"{query}::{dim}"
    if key not in _embed_cache:
        client = _get_openai_client()
        resp = client.embeddings.create(
            model=OPENAI_EMBED_MODEL, input=[query],
            encoding_format="float", dimensions=dim,
        )
        v = np.array([resp.data[0].embedding], dtype="float32")
        faiss.normalize_L2(v)
        _embed_cache[key] = v
    return _embed_cache[key].copy()


# ---------------------------------------------------------------------------
# Base rankings — computed once per question, reused across configurations
# ---------------------------------------------------------------------------

def base_rankings(query: str, index, chunks, bm25) -> dict:
    tokens = [
        w for w in word_tokenize(query.lower())
        if w.isalpha() and w not in stopset()
    ]
    bm25_scores = bm25.get_scores(tokens)
    bm25_order = [int(i) for i in np.argsort(bm25_scores)[::-1]]

    q_vec = embed(query, index.d)
    _, sem_idx = index.search(q_vec, len(chunks))
    dense_order = [int(i) for i in sem_idx[0] if i != -1]

    return {"bm25": bm25_order, "dense": dense_order}


def fuse(order_a: list[int], order_b: list[int], rrf_k: int = 60) -> dict[int, float]:
    rrf: dict[int, float] = {}
    for rank, idx in enumerate(order_a):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, idx in enumerate(order_b):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    return rrf


def dedup(order: list[int], chunks, limit: int) -> list[int]:
    seen: set[str] = set()
    out: list[int] = []
    for idx in order:
        t = chunks[idx]["text"].strip()
        if t not in seen:
            seen.add(t)
            out.append(idx)
        if len(out) >= limit:
            break
    return out


def run_config(
    base: dict,
    chunks,
    *,
    retriever: str,
    reranker: str,
    blend: float,
    candidates: int,
    k: int,
    query: str,
    max_per_source: int = 3,
    qid: str = "",
) -> list[int]:
    """Produce the final top-k for one configuration."""
    if retriever == "bm25":
        order, prior = base["bm25"], None
    elif retriever == "dense":
        order, prior = base["dense"], None
    else:
        rrf = fuse(base["bm25"], base["dense"])
        order = sorted(rrf, key=rrf.__getitem__, reverse=True)
        prior = rrf

    pool = dedup(order, chunks, candidates)

    if reranker == "none":
        ranked = pool
    else:
        key = (reranker, qid, len(pool), pool[0] if pool else -1,
               pool[-1] if pool else -1)
        if key not in _score_cache:
            ce = get_ce(reranker)
            _score_cache[key] = np.asarray(
                ce.predict([(query, chunks[i]["text"]) for i in pool]),
                dtype="float64",
            )
        scores = _score_cache[key]
        if blend > 0.0 and prior is not None:
            # Normalise both signals to [0,1] and blend, so the cross-encoder
            # refines the fusion ranking instead of overriding it outright.
            s = np.asarray(scores, dtype="float64")
            s = (s - s.min()) / (float(np.ptp(s)) or 1.0)
            p = np.array([prior.get(i, 0.0) for i in pool], dtype="float64")
            p = (p - p.min()) / (float(np.ptp(p)) or 1.0)
            combined = (1.0 - blend) * s + blend * p
        else:
            combined = np.asarray(scores, dtype="float64")
        ranked = [i for i, _ in sorted(zip(pool, combined),
                                       key=lambda x: x[1], reverse=True)]

    counts: dict[str, int] = {}
    final: list[int] = []
    for idx in ranked:
        src = chunks[idx]["source"]
        if counts.get(src, 0) < max_per_source:
            counts[src] = counts.get(src, 0) + 1
            final.append(idx)
        if len(final) >= k:
            break
    return final


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieval ablation sweep")
    ap.add_argument("--questions", default=str(DATA_DIR / "questions.json"))
    ap.add_argument("--output", default=str(DATA_DIR / "experiments.json"))
    ap.add_argument("--quick", action="store_true", help="small sweep, one reranker")
    ap.add_argument("--rerankers", nargs="*", default=None)
    args = ap.parse_args()

    items = [
        it for it in json.loads(Path(args.questions).read_text())
        if it.get("expected_answer")
        and "unanswerable" not in str(it.get("sub_type", ""))
        and "ambiguous" not in str(it.get("sub_type", ""))
    ]
    chunks, index, bm25 = load_artifacts(DATA_DIR)
    print(f"Loaded {len(chunks):,} chunks · {len(items)} answerable questions\n")

    # Gold sets, computed once.
    print("Computing gold chunk sets …", flush=True)
    gold: dict[str, set[int]] = {}
    for it in items:
        qid = it.get("id", it["question"][:20])
        gold[qid] = {
            i for i, c in enumerate(chunks)
            if content_hit(c["text"], it["expected_answer"])
        }

    print("Computing base rankings …", flush=True)
    bases: dict[str, dict] = {}
    for n, it in enumerate(items, 1):
        qid = it.get("id", it["question"][:20])
        bases[qid] = base_rankings(it["question"], index, chunks, bm25)
        print(f"  {n}/{len(items)}", end="\r", flush=True)
    print()

    rerankers = args.rerankers or (
        ["minilm-L6"] if args.quick else ["minilm-L6", "minilm-L12", "bge-base"]
    )

    configs = []
    # No-reranker baselines
    for retr in ["bm25", "dense", "hybrid"]:
        for k in ([8] if args.quick else [5, 8, 12]):
            configs.append(dict(retriever=retr, reranker="none", blend=0.0,
                                candidates=150, k=k))
    # Reranked variants, pure and blended
    for rr in rerankers:
        for cand in ([150] if args.quick else [50, 150]):
            for blend in [0.0, 0.3, 0.5]:
                for k in ([8] if args.quick else [5, 8, 12]):
                    configs.append(dict(retriever="hybrid", reranker=rr,
                                        blend=blend, candidates=cand, k=k))

    print(f"Running {len(configs)} configurations …\n")
    results = []
    for n, cfg in enumerate(configs, 1):
        t0 = time.perf_counter()
        hits = 0
        per_type: dict[str, list[bool]] = defaultdict(list)
        for it in items:
            qid = it.get("id", it["question"][:20])
            final = run_config(bases[qid], chunks, query=it["question"],
                               qid=qid, **cfg)
            hit = bool(set(final) & gold[qid])
            hits += hit
            per_type[it.get("sub_type", "untyped")].append(hit)

        recall = hits / len(items)
        row = {
            **cfg,
            "recall": round(recall, 4),
            "hits": hits,
            "n": len(items),
            "seconds": round(time.perf_counter() - t0, 1),
            "by_type": {t: round(sum(v) / len(v), 3) for t, v in per_type.items()},
        }
        results.append(row)
        name = (f"{cfg['retriever']}/{cfg['reranker']}"
                f"{'+blend' + str(cfg['blend']) if cfg['blend'] else ''}"
                f"/c{cfg['candidates']}/k{cfg['k']}")
        print(f"[{n:2d}/{len(configs)}] {name:<46} recall={recall:.1%} "
              f"({hits}/{len(items)})")

    results.sort(key=lambda r: (-r["recall"], r["k"]))

    print("\n" + "=" * 84)
    print("ABLATION RESULTS — ranked by recall@k")
    print("=" * 84)
    print(f"  {'retriever':<10} {'reranker':<12} {'blend':>6} {'cand':>5} "
          f"{'k':>3} {'recall':>8}  {'hits':>7}")
    print("  " + "-" * 80)
    for r in results[:20]:
        print(f"  {r['retriever']:<10} {r['reranker']:<12} {r['blend']:>6.1f} "
              f"{r['candidates']:>5} {r['k']:>3} {r['recall']:>7.1%}  "
              f"{r['hits']:>3}/{r['n']}")

    best = results[0]
    baseline = next(
        (r for r in results if r["retriever"] == "hybrid"
         and r["reranker"] == "minilm-L6" and r["blend"] == 0.0
         and r["candidates"] == 150 and r["k"] == 8),
        None,
    )
    print("\n" + "=" * 84)
    if baseline:
        print(f"  Incumbent config  hybrid/minilm-L6/c150/k8 : "
              f"{baseline['recall']:.1%}")
    print(f"  Best config       {best['retriever']}/{best['reranker']}"
          f"/blend{best['blend']}/c{best['candidates']}/k{best['k']} : "
          f"{best['recall']:.1%}")
    if baseline:
        delta = best["recall"] - baseline["recall"]
        print(f"  Improvement       {delta:+.1%}")
    print("=" * 84)

    out = Path(args.output)
    out.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
