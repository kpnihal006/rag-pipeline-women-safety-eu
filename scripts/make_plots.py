from __future__ import annotations

"""
scripts/make_plots.py

Render the figures used in the scientific report from the experiment JSON.

Every figure is regenerated from the saved result files, so the report cannot
drift from the measurements: change an experiment, re-run this, and the plots
follow. Nothing here calls a model or a network.

Figures produced in reports/figures/:
    fig1_failure_stages.png   where retrieval recall is lost, stage by stage
    fig2_retriever.png        retriever ablation (bm25 / dense / hybrid / +CE)
    fig3_chunking.png         chunk size and overlap, recall and MRR
    fig4_embedders.png        embedding model comparison (dense only)
    fig5_k_tradeoff.png       recall vs precision as k grows
    fig6_eval_categories.png  LLM-as-judge outcome by question category

Usage:
    uv run python -m scripts.make_plots
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(os.environ.get("PDF_OUTPUT_DIR", "data"))
FIG_DIR = Path("reports/figures")

# A single restrained palette so the figures read as one set.
INK = "#1d1d1f"
MUTED = "#86868b"
GRID = "#e5e5ea"
ACCENT = "#0b6bcb"
WARN = "#c1121f"
OK = "#2a9d5c"
ALT = "#9a6dd7"


def _style(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_title(title, fontsize=12, color=INK, pad=12, loc="left")
    ax.set_xlabel(xlabel, fontsize=10, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=10, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ---------------------------------------------------------------------------

def fig_failure_stages():
    d = _load(DATA_DIR / "retrieval_diagnosis.json")
    if not d:
        return
    rows = d["rows"]
    n = len(rows)
    stages = [
        ("in corpus", sum(r["content_in_corpus"] for r in rows)),
        ("survives RRF", sum(r["in_candidates"] for r in rows)),
        ("survives cross-encoder", sum(r["in_reranked_topk"] for r in rows)),
        ("survives diversity cap", sum(r["in_final"] for r in rows)),
    ]
    labels = [s[0] for s in stages]
    vals = [100 * s[1] / n for s in stages]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = [ACCENT, ACCENT, WARN, WARN]
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    for b, v, (_, c) in zip(bars, vals, stages):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5,
                f"{v:.1f}%\n({c}/{n})", ha="center", fontsize=9, color=INK)
    ax.set_ylim(0, 108)
    _style(ax, "Where retrieval recall is lost", ylabel="questions retaining gold (%)")
    ax.annotate("", xy=(2, vals[2] + 8), xytext=(1, vals[1] + 8),
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.6))
    ax.text(1.5, vals[1] + 12, f"-{vals[1]-vals[2]:.1f} pts", ha="center",
            color=WARN, fontsize=10, fontweight="bold")
    plt.xticks(rotation=12, ha="right")
    _save(fig, "fig1_failure_stages.png")


def fig_retriever():
    d = _load(DATA_DIR / "retrieval_diagnosis.json")
    if not d:
        return
    rows = d["rows"]
    n = len(rows)
    data = [
        ("BM25 only", 100 * sum(r["bm25_only_hit"] for r in rows) / n),
        ("Dense only", 100 * sum(r["dense_only_hit"] for r in rows) / n),
        ("Hybrid + cross-encoder\n(production)",
         100 * sum(r["in_final"] for r in rows) / n),
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [MUTED, OK, WARN]
    bars = ax.barh([d[0] for d in data], [d[1] for d in data],
                   color=colors, height=0.55)
    for b, (_, v) in zip(bars, data):
        ax.text(v + 1.2, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
                va="center", fontsize=10, color=INK)
    ax.set_xlim(0, 100)
    _style(ax, "Retriever ablation — the full pipeline scores below dense alone",
           xlabel="recall (%)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    _save(fig, "fig2_retriever.png")


def fig_chunking():
    d = _load(DATA_DIR / "experiment_grid.json")
    if not d:
        return
    res = [r for r in d["results"] if r["retriever"] == "dense" and r["k"] == 8]
    if not res:
        return

    sizes = sorted({r["chunk_size"] for r in res})
    overlaps = sorted({r["overlap"] for r in res})

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for i, ov in enumerate(overlaps):
        ys = [100 * max((r["recall"] for r in res
                         if r["chunk_size"] == s and r["overlap"] == ov),
                        default=0) for s in sizes]
        a1.plot(sizes, ys, marker="o", label=f"overlap {ov}",
                color=[ACCENT, OK, ALT][i % 3], linewidth=2)
    _style(a1, "Chunk size × overlap — recall",
           xlabel="chunk size (chars)", ylabel="recall@8 (%)")
    a1.legend(frameon=False, fontsize=9)

    for i, ov in enumerate(overlaps):
        ys = [max((r["mrr"] for r in res
                   if r["chunk_size"] == s and r["overlap"] == ov), default=0)
              for s in sizes]
        a2.plot(sizes, ys, marker="s", label=f"overlap {ov}",
                color=[ACCENT, OK, ALT][i % 3], linewidth=2)
    _style(a2, "Chunk size × overlap — ranking quality (MRR)",
           xlabel="chunk size (chars)", ylabel="MRR@8")
    a2.legend(frameon=False, fontsize=9)

    fig.suptitle("Overlap does not change what is found, only where it ranks",
                 fontsize=10, color=MUTED, y=1.02, x=0.01, ha="left")
    _save(fig, "fig3_chunking.png")


def fig_embedders():
    d = _load(DATA_DIR / "exp_embedding.json")
    if not d:
        return
    res = [r for r in d["results"] if r["retriever"] == "dense" and r["k"] == 8]
    if not res:
        return
    res.sort(key=lambda r: -r["mrr"])
    names = [f"{r['embedder']}\n({r['dim']}d)" for r in res]
    x = range(len(res))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    w = 0.38
    ax.bar([i - w / 2 for i in x], [100 * r["recall"] for r in res],
           width=w, label="recall@8 (%)", color=ACCENT)
    ax.bar([i + w / 2 for i in x], [100 * r["mrr"] for r in res],
           width=w, label="MRR@8 ×100", color=ALT)
    for i, r in enumerate(res):
        ax.text(i - w / 2, 100 * r["recall"] + 1, f"{r['recall']:.1%}",
                ha="center", fontsize=9, color=INK)
        ax.text(i + w / 2, 100 * r["mrr"] + 1, f"{r['mrr']:.3f}",
                ha="center", fontsize=9, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    _style(ax, "Embedding models — dense retrieval only (BM25 held out)")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, "fig4_embedders.png")


def fig_k_tradeoff():
    d = _load(DATA_DIR / "exp_k.json")
    if not d:
        return
    # Retrievers behave differently with k, so plot them separately rather
    # than collapsing to a max that hides the difference.
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    styles = {"dense": (ACCENT, "o"), "bm25": (MUTED, "^"), "hybrid": (ALT, "D")}
    ks = sorted({r["k"] for r in d["results"]})
    for retr, (color, marker) in styles.items():
        rows = {r["k"]: r for r in d["results"] if r["retriever"] == retr}
        if not rows:
            continue
        ax.plot(ks, [100 * rows[k]["recall"] for k in ks], marker=marker,
                color=color, linewidth=2, label=f"{retr} — recall@k")
    rows = {r["k"]: r for r in d["results"] if r["retriever"] == "dense"}
    ax.plot(ks, [100 * rows[k]["precision"] for k in ks], marker="s",
            color=WARN, linewidth=2, linestyle="--", label="dense — precision@k")
    _style(ax, "Recall gain from larger k is dense-only, and precision pays for it",
           xlabel="k (passages retrieved)", ylabel="%")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xticks(ks)
    _save(fig, "fig5_k_tradeoff.png")


def fig_eval_categories():
    d = _load(DATA_DIR / "eval_results.json")
    if not d:
        return
    rows = d.get("results", [])
    cats: dict[str, list[bool]] = {}
    for r in rows:
        c = r.get("sub_type") or r.get("category") or "untyped"
        v = r.get("judge_verdict")
        if v in ("PASS", "FAIL"):
            cats.setdefault(c, []).append(v == "PASS")
    if not cats:
        return
    items = sorted(cats.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
    labels = [f"{k}  (n={len(v)})" for k, v in items]
    vals = [100 * sum(v) / len(v) for _, v in items]

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.42 * len(items) + 1.5)))
    colors = [WARN if v < 50 else (ACCENT if v < 100 else OK) for v in vals]
    bars = ax.barh(labels, vals, color=colors, height=0.6)
    for b, v in zip(bars, vals):
        ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.0f}%",
                va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 108)
    _style(ax, "LLM-as-judge pass rate by question category", xlabel="pass rate (%)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    _save(fig, "fig6_eval_categories.png")


def main() -> None:
    print("Rendering figures …")
    for fn in (fig_failure_stages, fig_retriever, fig_chunking,
               fig_embedders, fig_k_tradeoff, fig_eval_categories):
        try:
            fn()
        except Exception as exc:
            print(f"  {fn.__name__}: skipped ({exc})")
    print(f"\nFigures → {FIG_DIR}/")


if __name__ == "__main__":
    main()
