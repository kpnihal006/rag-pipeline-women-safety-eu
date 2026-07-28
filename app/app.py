from __future__ import annotations

import json
import os
import sys
import tempfile
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
from dotenv import load_dotenv

from scripts.chunk import generate_answer, load_artifacts, retrieve
from app.mcp_server import add_to_database, ingest_pdf

load_dotenv()

DATA_DIR      = Path(os.environ.get("PDF_OUTPUT_DIR", "data"))
PDF_INPUT_DIR = Path(os.environ.get("PDF_INPUT_DIR", "data/pdfs"))
FEEDBACK_FILE = DATA_DIR / "feedback.json"
QUERIES_FILE  = DATA_DIR / "queries.json"

_chunks = _index = _bm25 = None


def _get_artifacts():
    global _chunks, _index, _bm25
    if _chunks is None:
        _chunks, _index, _bm25 = load_artifacts(DATA_DIR)
    return _chunks, _index, _bm25


def _reload_artifacts() -> None:
    global _chunks, _index, _bm25
    _chunks = _index = _bm25 = None


def _corpus_stats() -> tuple[int, int]:
    try:
        chunks, _, _ = _get_artifacts()
        return len(chunks), len({c["source"] for c in chunks})
    except Exception:
        return 0, 0


CONFIDENCE_THRESHOLD = 0.50
_NO_CITATION_PHRASES = (
    "outside the scope",
    "not available in the corpus",
    "too broad to answer",
)

# ---------------------------------------------------------------------------
# Query logging
# ---------------------------------------------------------------------------

def _log_query(query: str, top_score: float, n_results: int) -> None:
    data: list = []
    if QUERIES_FILE.exists():
        try:
            data = json.loads(QUERIES_FILE.read_text())
        except Exception:
            data = []
    data.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "top_score": round(top_score, 4),
        "n_results": n_results,
    })
    QUERIES_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _load_queries() -> list[dict]:
    if not QUERIES_FILE.exists():
        return []
    try:
        return json.loads(QUERIES_FILE.read_text())
    except Exception:
        return []


def _load_feedback() -> list[dict]:
    if not FEEDBACK_FILE.exists():
        return []
    try:
        return json.loads(FEEDBACK_FILE.read_text())
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

def _append_feedback(question: str, answer: str, verdict: str) -> None:
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = _load_feedback()
    data.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer[:400],
        "feedback": verdict,
    })
    FEEDBACK_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _extract_text(value) -> str:
    """Reliably extract plain text from a Gradio message value (str, dict, or list)."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("text") or value.get("content") or value.get("value") or str(value)
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text") or item.get("content") or item.get("value") or "")
        return " ".join(p for p in parts if p)
    return str(value)


def handle_like(data: gr.LikeData, history: list) -> None:
    if not history:
        return
    verdict = "helpful" if data.liked else "not_helpful"
    liked_content = _extract_text(data.value)
    liked_idx = data.index if isinstance(data.index, int) else (data.index[0] if data.index else -1)
    question = ""
    for msg in reversed(history[:liked_idx]):
        if msg.get("role") == "user":
            question = _extract_text(msg.get("content", ""))
            break
    _append_feedback(question, liked_content, verdict)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_conversation(history: list):
    if not history:
        return gr.update(value=None, visible=False)
    lines = ["Women's Safety Laws — RAG Conversation Export", "=" * 52, ""]
    for msg in history:
        role = "You" if msg.get("role") == "user" else "Assistant"
        raw = msg.get("content", "")
        text = raw if isinstance(raw, str) else " ".join(str(c) for c in raw)
        lines += [f"{role}:", text, ""]
    content = "\n".join(lines)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False,
        encoding="utf-8", prefix="rag_export_",
    )
    tmp.write(content)
    tmp.close()
    return gr.update(value=tmp.name, visible=True)


# ---------------------------------------------------------------------------
# Sources: corpus HTML + PDF links
# ---------------------------------------------------------------------------

def _score_colour(score: float, threshold: float) -> str:
    if score >= threshold + 0.15:
        return "#34C759"   # Apple green
    if score >= threshold:
        return "#FF9F0A"   # Apple orange
    return "#FF3B30"       # Apple red


def _build_corpus_html(results: list[dict], answer: str, threshold: float) -> str:
    if any(p in answer.lower() for p in _NO_CITATION_PHRASES) or not results:
        return "<p style='color:#6B7280;font-style:italic;padding:8px'>No corpus sources retrieved.</p>"

    top_score = max(r["score"] for r in results)
    warning = ""
    if top_score < threshold:
        warning = (
            "<div style='background:#FFF8EC;border:1px solid #FFDCA8;border-radius:12px;"
            "padding:10px 14px;margin-bottom:12px;font-size:.83em;"
            "font-family:-apple-system,system-ui,sans-serif;color:#3A3A3C'>"
            f"⚠️ <b style='color:#FF9F0A'>Low confidence</b> — top score {top_score:.2f} "
            f"(threshold {threshold:.2f}). Answer may be incomplete.</div>"
        )

    cards = []
    for i, r in enumerate(results, 1):
        colour  = _score_colour(r["score"], threshold)
        preview = r["text"][:280].replace("<", "&lt;").replace(">", "&gt;")
        if len(r["text"]) > 280:
            preview += "…"

        pdf_abs  = (PDF_INPUT_DIR / r["source"]).resolve()
        pdf_link = ""
        if pdf_abs.exists():
            pdf_url = f"/file={pdf_abs}#page={r['page']}"
            pdf_link = (
                f"<a href='{pdf_url}' target='_blank' rel='noopener' "
                f"style='font-size:.74em;color:#0071E3;font-weight:600;"
                f"text-decoration:none;border:1px solid #B3D4F5;border-radius:980px;"
                f"padding:2px 9px;margin-left:6px;white-space:nowrap'>"
                f"📄 p.{r['page']}</a>"
            )

        cards.append(f"""
<div style='border:1px solid #D2D2D7;border-radius:14px;padding:12px 15px;
            margin-bottom:9px;background:#FFFFFF;font-size:.84em;
            font-family:-apple-system,system-ui,sans-serif'>
  <div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:7px'>
    <span style='font-weight:700;color:#6E6E73;font-size:.8em'>#{i}</span>
    <span style='flex:1;font-weight:600;color:#1D1D1F;word-break:break-word'>{r["source"]}</span>
    <span style='background:#F5F5F7;border-radius:6px;padding:2px 7px;font-size:.78em;
                 color:#6E6E73'>p.{r["page"]}</span>
    <span style='background:{colour}18;color:{colour};border-radius:980px;
                 padding:2px 8px;font-weight:700;font-size:.78em;
                 border:1px solid {colour}44'>{r["score"]:.3f}</span>
    {pdf_link}
  </div>
  <div style='color:#3A3A3C;line-height:1.55;border-left:2px solid {colour};
              padding-left:10px;font-size:.92em'>{preview}</div>
</div>""")

    return warning + "\n".join(cards)


# ---------------------------------------------------------------------------
# Sources: web search HTML
# ---------------------------------------------------------------------------

def _fetch_web_sources(query: str, max_results: int = 5) -> str:
    """Run a DuckDuckGo search and return styled HTML cards with clickable links."""
    try:
        from duckduckgo_search import DDGS
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = list(DDGS().text(query + " EU law women rights", max_results=max_results))
        if not results:
            return "<p style='color:#6B7280;font-style:italic;padding:8px'>No web results found.</p>"

        cards = []
        for r in results:
            title   = (r.get("title") or "Untitled")[:90]
            href    = r.get("href") or "#"
            body    = (r.get("body") or "")[:200].replace("<", "&lt;").replace(">", "&gt;")
            domain  = href.split("/")[2] if "/" in href else href

            cards.append(f"""
<div style='border:1px solid #D2D2D7;border-radius:14px;padding:12px 15px;
            margin-bottom:9px;background:#FFFFFF;font-size:.84em;
            font-family:-apple-system,system-ui,sans-serif;
            transition:box-shadow .15s'
     onmouseover="this.style.boxShadow='0 4px 16px rgba(0,113,227,.12)'"
     onmouseout="this.style.boxShadow='none'">
  <a href='{href}' target='_blank' rel='noopener noreferrer'
     style='color:#0071E3;font-weight:600;text-decoration:none;line-height:1.35;
            display:block;margin-bottom:5px'>{title} ↗</a>
  <div style='color:#3A3A3C;line-height:1.55'>{body}{'…' if body else ''}</div>
  <div style='color:#8E8E93;font-size:.74em;margin-top:6px;word-break:break-all'>{domain}</div>
</div>""")

        return (
            "<div style='font-size:.72em;font-weight:600;color:#6E6E73;text-transform:uppercase;"
            "letter-spacing:.06em;margin-bottom:8px;"
            "font-family:-apple-system,system-ui,sans-serif'>🌐 Live Web Sources</div>"
            + "\n".join(cards)
        )
    except Exception as exc:
        return f"<p style='color:#9CA3AF;font-size:.8em'>Web search unavailable: {exc}</p>"


# ---------------------------------------------------------------------------
# PDF page rendering
# ---------------------------------------------------------------------------

def _render_pdf_pages(results: list[dict]) -> list[tuple]:
    seen: set = set()
    images: list[tuple] = []
    for r in results:
        key = (r["source"], r["page"])
        if key in seen:
            continue
        seen.add(key)
        pdf_path = PDF_INPUT_DIR / r["source"]
        if not pdf_path.exists():
            continue
        try:
            doc = fitz.open(str(pdf_path))
            page_idx = r["page"] - 1
            if page_idx < 0 or page_idx >= len(doc):
                doc.close()
                continue
            pix = doc.load_page(page_idx).get_pixmap(matrix=fitz.Matrix(2, 2))
            doc.close()
            tmp = tempfile.NamedTemporaryFile(
                suffix=".png", delete=False,
                dir=tempfile.gettempdir(),
                prefix=f"rag_p{r['page']}_",
            )
            pix.save(tmp.name)
            tmp.close()
            images.append((tmp.name, f"{r['source']}  p.{r['page']}  ({r['score']:.3f})"))
        except Exception:
            pass
    return images


# ---------------------------------------------------------------------------
# Format answer text for chat
# ---------------------------------------------------------------------------

def _format_answer(answer: str, results: list[dict]) -> str:
    if any(p in answer.lower() for p in _NO_CITATION_PHRASES) or not results:
        return answer
    seen, lines = set(), []
    for r in results:
        key = (r["source"], r["page"])
        if key not in seen:
            seen.add(key)
            lines.append(f"- **{r['source']}** — Page {r['page']}")
    return f"{answer}\n\n---\n**Sources**\n" + "\n".join(lines) if lines else answer


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------

def _stats_html() -> str:
    try:
        n_chunks, n_docs = _corpus_stats()
    except Exception:
        n_chunks, n_docs = "—", "—"
    return f"""
<div style="display:flex;gap:10px;flex-wrap:wrap;margin:4px 0;
            font-family:-apple-system,system-ui,sans-serif">
  <div style="background:#F5F5F7;border:1px solid #D2D2D7;border-radius:14px;
              padding:12px 18px;flex:1;min-width:100px;">
    <div style="font-size:1.65em;font-weight:700;color:#0071E3;line-height:1;
                letter-spacing:-.02em">{n_chunks:,}</div>
    <div style="color:#6E6E73;font-size:.77em;font-weight:500;margin-top:3px;
                text-transform:uppercase;letter-spacing:.05em">chunks indexed</div>
  </div>
  <div style="background:#F5F5F7;border:1px solid #D2D2D7;border-radius:14px;
              padding:12px 18px;flex:1;min-width:100px;">
    <div style="font-size:1.65em;font-weight:700;color:#34C759;line-height:1;
                letter-spacing:-.02em">{n_docs}</div>
    <div style="color:#6E6E73;font-size:.77em;font-weight:500;margin-top:3px;
                text-transform:uppercase;letter-spacing:.05em">source docs</div>
  </div>
</div>"""


def handle_add_text(text: str) -> tuple[str, str]:
    if not text.strip():
        return "Please enter some text to add.", _stats_html()
    try:
        from datetime import date
        result = add_to_database(text=text.strip(), source=f"ui-{date.today().isoformat()}")
        _reload_artifacts()
        return f"Done. {result}", _stats_html()
    except Exception as exc:
        return f"Error: {exc}", _stats_html()


def handle_ingest_pdf(file):
    if file is None:
        yield "No file uploaded.", _stats_html()
        return
    pdf_path = file if isinstance(file, str) else file.name
    yield _spinner("Extracting and embedding PDF…"), _stats_html()
    try:
        result = ingest_pdf(pdf_path=pdf_path)
        _reload_artifacts()
        yield f"Done. {result}", _stats_html()
    except Exception as exc:
        yield f"Error: {exc}", _stats_html()


# ---------------------------------------------------------------------------
# HITL helpers
# ---------------------------------------------------------------------------

def approve_last_answer(history: list) -> str:
    if not history:
        return "No conversation to approve."
    last_assistant = last_user = None
    for msg in reversed(history):
        role = msg.get("role", "")
        content = _extract_text(msg.get("content", ""))
        if role == "assistant" and last_assistant is None:
            last_assistant = content
        elif role == "user" and last_user is None and last_assistant is not None:
            last_user = content
            break
    if not last_assistant:
        return "No answer to approve."
    try:
        reports_dir = DATA_DIR / "approved"
        reports_dir.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (last_user or "answer")[:40])
        path = reports_dir / f"{ts}_{safe}.md"
        path.write_text(f"# {last_user or 'Approved Answer'}\n\n{last_assistant}\n", encoding="utf-8")
        _append_feedback(last_user or "", last_assistant, "approved")
        return f"✅ Approved — saved to {path.name}"
    except Exception as exc:
        return f"❌ Error saving: {exc}"


def reject_last_answer(history: list) -> tuple[list, str]:
    if not history:
        return history, "Nothing to reject."
    new_h = list(history)
    if new_h and new_h[-1]["role"] == "assistant":
        new_h.pop()
    if new_h and new_h[-1]["role"] == "user":
        new_h.pop()
    return new_h, "🗑️ Answer rejected and removed."


def request_rewrite(feedback: str, history: list) -> tuple[list, str]:
    if not feedback.strip():
        return history, "Please enter feedback describing what to change."
    if not history:
        return history, "No history to rewrite."
    idx = next((i for i in reversed(range(len(history))) if history[i]["role"] == "assistant"), None)
    if idx is None:
        return history, "No assistant message found."
    current = history[idx].get("content", "")
    query = next(
        (_extract_text(history[i].get("content", "")) for i in reversed(range(idx)) if history[i]["role"] == "user"), "",
    )
    # Routed through app.llm so the rewrite runs on the local backend like the
    # rest of the pipeline; a hardcoded hosted model here would silently break
    # the "local models only" constraint from the Gradio UI.
    from app import llm as _llm
    resp = _llm.get_client().chat.completions.create(
        model=_llm.chat_model(), max_tokens=2048,
        messages=[
            {"role": "system", "content": "Rewrite the answer based on feedback. Keep it factual and cited."},
            {"role": "user", "content": f"Question: {query}\n\nCurrent answer:\n{current}\n\nFeedback: {feedback}"},
        ],
    )
    new_answer = resp.choices[0].message.content or current
    return (
        history[:idx] + [{"role": "assistant", "content": new_answer}] + history[idx + 1:],
        "✏️ Answer rewritten.",
    )


# ---------------------------------------------------------------------------
# Analytics charts
# ---------------------------------------------------------------------------

_STYLE = {
    "bg":     "#FBFBFD",   # Apple page background
    "blue":   "#0071E3",   # Apple accent blue
    "green":  "#34C759",   # Apple green
    "amber":  "#FF9F0A",   # Apple orange
    "red":    "#FF3B30",   # Apple red
    "grid":   "#E8E8ED",   # Apple divider
    "text":   "#1D1D1F",   # Apple primary text
    "sec":    "#6E6E73",   # Apple secondary text
    "gold":   "#F5A623",   # warm accent
}


def _apply_style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(_STYLE["bg"])
    ax.figure.patch.set_facecolor(_STYLE["bg"])
    ax.set_title(title, fontsize=11, fontweight="bold", color=_STYLE["text"], pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=_STYLE["text"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=_STYLE["text"])
    ax.tick_params(colors=_STYLE["text"], labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_STYLE["grid"])
    ax.yaxis.grid(True, color=_STYLE["grid"], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def _no_data_fig(label: str):
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor(_STYLE["bg"])
    ax.set_facecolor(_STYLE["bg"])
    ax.text(0.5, 0.5, f"No data yet\n({label})", ha="center", va="center",
            transform=ax.transAxes, color="#9CA3AF", fontsize=10)
    ax.axis("off")
    return fig


def _chart_top_questions(queries: list[dict]):
    if not queries:
        return _no_data_fig("queries")
    counter = Counter(q["query"][:55] for q in queries).most_common(10)
    labels = [lbl for lbl, _ in reversed(counter)]
    values = [cnt for _, cnt in reversed(counter)]
    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.55)))
    bars = ax.barh(labels, values, color=_STYLE["blue"], alpha=0.85, height=0.6, zorder=3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8, color=_STYLE["text"])
    _apply_style(ax, "Top Questions", xlabel="Times asked")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    return fig


def _chart_daily_usage(queries: list[dict]):
    if not queries:
        return _no_data_fig("usage")
    daily: Counter = Counter()
    for q in queries:
        daily[q["timestamp"][:10]] += 1
    days = sorted(daily)
    counts = [daily[d] for d in days]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(range(len(days)), counts, alpha=0.18, color=_STYLE["blue"])
    ax.plot(range(len(days)), counts, marker="o", markersize=5,
            color=_STYLE["blue"], linewidth=1.8, zorder=3)
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(days, rotation=45, ha="right", fontsize=7)
    _apply_style(ax, "Daily Query Volume", ylabel="Queries")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    return fig


def _chart_feedback_pie(feedback: list[dict]):
    if not feedback:
        return _no_data_fig("feedback")
    counts = Counter(f["feedback"] for f in feedback)
    colour_map = {
        "helpful":     _STYLE["green"],
        "not_helpful": _STYLE["red"],
        "approved":    _STYLE["blue"],
    }
    labels = list(counts.keys())
    sizes  = list(counts.values())
    colors = [colour_map.get(l, _STYLE["amber"]) for l in labels]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.0f%%", colors=colors,
        startangle=90, wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
        pctdistance=0.78, textprops={"fontsize": 9, "color": _STYLE["text"]},
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("white")
        at.set_fontweight("bold")
    ax.set_title("Feedback Distribution", fontsize=11, fontweight="bold",
                 color=_STYLE["text"], pad=10)
    fig.patch.set_facecolor(_STYLE["bg"])
    plt.tight_layout()
    return fig


def _chart_confidence_hist(queries: list[dict]):
    scores = [q["top_score"] for q in queries if q.get("top_score", 0) > 0]
    if not scores:
        return _no_data_fig("confidence")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(scores, bins=20, color=_STYLE["blue"],
            edgecolor="white", linewidth=0.5, alpha=0.85, zorder=3)
    threshold = CONFIDENCE_THRESHOLD
    ax.axvline(threshold, color=_STYLE["red"], linestyle="--",
               linewidth=1.4, label=f"threshold ({threshold})", zorder=4)
    ax.legend(fontsize=8, framealpha=0.9)
    _apply_style(ax, "Retrieval Confidence Distribution",
                 xlabel="Top score", ylabel="Queries")
    plt.tight_layout()
    return fig


def _analytics_summary_html(queries: list[dict], feedback: list[dict]) -> str:
    n_total   = len(queries)
    avg_score = (sum(q.get("top_score", 0) for q in queries) / n_total) if n_total else 0
    n_helpful = sum(1 for f in feedback if f.get("feedback") == "helpful")
    n_fb      = len(feedback)
    approval  = f"{n_helpful}/{n_fb}" if n_fb else "—"
    low_conf  = sum(1 for q in queries if q.get("top_score", 1) < CONFIDENCE_THRESHOLD)

    def card(accent, value, label):
        return (
            f"<div style='background:#F5F5F7;border:1px solid #D2D2D7;border-radius:14px;"
            f"padding:12px 16px;flex:1;min-width:80px;'>"
            f"<div style='font-size:1.55em;font-weight:700;color:{accent};"
            f"line-height:1;letter-spacing:-.02em'>{value}</div>"
            f"<div style='color:#6E6E73;font-size:.73em;font-weight:500;margin-top:4px;"
            f"text-transform:uppercase;letter-spacing:.05em'>{label}</div>"
            f"</div>"
        )

    score_color = "#0071E3" if avg_score >= CONFIDENCE_THRESHOLD else "#FF9F0A"
    return (
        "<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;"
        "font-family:-apple-system,system-ui,sans-serif'>"
        + card("#0071E3", f"{n_total:,}", "total queries")
        + card(score_color,   f"{avg_score:.2f}", "avg confidence")
        + card("#34C759", approval,         "helpful feedback")
        + card("#FF3B30", str(low_conf),    "low confidence")
        + "</div>"
    )


def refresh_analytics():
    queries  = _load_queries()
    feedback = _load_feedback()
    summary  = _analytics_summary_html(queries, feedback)
    fig1 = _chart_top_questions(queries)
    fig2 = _chart_daily_usage(queries)
    fig3 = _chart_feedback_pie(feedback)
    fig4 = _chart_confidence_hist(queries)

    recent = [
        [
            q["timestamp"][:16].replace("T", " "),
            q["query"][:70],
            round(q.get("top_score", 0), 3),
        ]
        for q in reversed(queries[-20:])
    ]
    return summary, fig1, fig2, fig3, fig4, recent


# ---------------------------------------------------------------------------
# Feedback list renderer
# ---------------------------------------------------------------------------

def _render_feedback_html(entries: list[dict]) -> str:
    if not entries:
        return (
            "<div style='padding:20px;text-align:center;color:#8E8E93;"
            "font-size:.9em;font-family:-apple-system,system-ui,sans-serif'>"
            "No feedback recorded yet.</div>"
        )
    _badge = {
        "helpful":     ("#34C759", "👍 Helpful"),
        "not_helpful": ("#FF3B30", "👎 Not helpful"),
        "approved":    ("#0071E3", "✅ Approved"),
    }
    rows = []
    for e in entries:
        colour, label = _badge.get(e.get("feedback", ""), ("#8E8E93", e.get("feedback", "—")))
        raw_q = e.get("question") or "—"
        if not isinstance(raw_q, str):
            raw_q = str(raw_q)
        question  = raw_q[:100]
        q_ellipsis = "…" if len(raw_q) > 100 else ""

        raw_a = e.get("answer") or ""
        if not isinstance(raw_a, str):
            raw_a = str(raw_a)
        # Strip markdown source block appended by _format_answer
        raw_a = raw_a.split("\n\n---\n")[0].strip()
        answer    = raw_a[:180]
        a_ellipsis = "…" if len(raw_a) > 180 else ""

        rows.append(
            f"<div style='padding:10px 12px;border-bottom:1px solid #F5F5F7;"
            f"font-family:-apple-system,system-ui,sans-serif'>"
            f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:5px'>"
            f"<span style='background:{colour}18;color:{colour};border:1px solid {colour}44;"
            f"border-radius:980px;padding:2px 10px;font-size:.73em;font-weight:600;"
            f"white-space:nowrap;flex-shrink:0'>{label}</span>"
            f"<span style='color:#1D1D1F;font-size:.84em;font-weight:600;line-height:1.4'>"
            f"{question}{q_ellipsis}</span>"
            f"</div>"
            + (
                f"<div style='color:#6E6E73;font-size:.8em;line-height:1.5;"
                f"padding-left:2px'>{answer}{a_ellipsis}</div>"
                if answer else ""
            )
            + "</div>"
        )
    header = (
        "<div style='font-size:.72em;font-weight:600;color:#6E6E73;text-transform:uppercase;"
        "letter-spacing:.06em;padding:8px 12px 4px;"
        "font-family:-apple-system,system-ui,sans-serif'>Recent Feedback</div>"
    )
    container_open  = "<div style='background:#FFFFFF;border:1px solid #D2D2D7;border-radius:14px;overflow:hidden'>"
    container_close = "</div>"
    return header + container_open + "\n".join(rows) + container_close


# ---------------------------------------------------------------------------
# Spinner
# ---------------------------------------------------------------------------

def _spinner(label: str) -> str:
    return (
        "<div style='display:flex;align-items:center;gap:8px;color:#0071E3;"
        "font-size:.88em;font-weight:500;padding:4px 0;"
        "font-family:-apple-system,system-ui,sans-serif'>"
        "<svg width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='currentColor'"
        " stroke-width='2.5' stroke-linecap='round'"
        " style='animation:rag-spin 1s linear infinite'>"
        "<path d='M21 12a9 9 0 1 1-6.219-8.56'/></svg>"
        f"{label}</div>"
        "<style>@keyframes rag-spin{{to{{transform:rotate(360deg)}}}}</style>"
    )


# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------

def _raw_web_results(query: str, max_results: int = 5) -> list[dict]:
    """Fetch DuckDuckGo snippets; return list of {title, href, body}."""
    try:
        from duckduckgo_search import DDGS
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return list(DDGS().text(query + " EU law women rights", max_results=max_results))
    except Exception:
        return []


def _generate_answer_web_augmented(
    query: str,
    corpus_results: list[dict],
    web_results: list[dict],
    history: list | None,
    user: str | None,
) -> str:
    """Call the LLM with both corpus excerpts AND live web snippets.

    Unlike generate_answer(), this prompt explicitly allows the model to draw
    on web sources when the corpus has insufficient coverage.
    """
    import openai as _oa
    from scripts.chunk import OPENAI_MODEL, _get_openai_client, track_cost

    corpus_ctx = "\n\n---\n\n".join(
        f"[CORPUS: {r['source']} p.{r['page']}]\n{r['text']}"
        for r in corpus_results
    ) or "(no corpus excerpts retrieved)"

    web_ctx = "\n\n---\n\n".join(
        f"[WEB: {r.get('title','')[:60]}] {r.get('href','')}\n{r.get('body','')}"
        for r in web_results
        if r.get("body")
    ) or "(no web results)"

    system = (
        "You are a legal research assistant specialising in EU women's safety laws, "
        "gender equality, and human rights.\n\n"
        "You have access to two sources of information:\n"
        "1. CORPUS EXCERPTS — authoritative EU legal texts and EIGE reports.\n"
        "2. WEB SNIPPETS — live search results from the public web.\n\n"
        "Answer rules (in priority order):\n"
        "- If CORPUS EXCERPTS contain relevant information, use them first and cite as [CORPUS: filename p.N].\n"
        "- If corpus is insufficient, use WEB SNIPPETS and cite as [WEB: domain].\n"
        "- If neither source covers the question but you have reliable general knowledge, "
        "use it and note that it comes from your training knowledge.\n"
        "- Only decline to answer if the question is completely unrelated to women's rights, "
        "gender equality, legal matters, or general knowledge.\n"
        "- Be concise, factual, and accurate.\n\n"
        f"CORPUS EXCERPTS:\n{corpus_ctx}\n\n"
        f"WEB SNIPPETS:\n{web_ctx}"
    )

    messages: list[dict] = [{"role": "system", "content": system}]
    if history:
        for turn in (history or [])[-10:]:
            if isinstance(turn, dict):
                messages.append({"role": turn["role"], "content": turn["content"]})
            else:
                u, a = turn
                messages.append({"role": "user", "content": u})
                messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": query})

    client   = _get_openai_client()
    response = client.chat.completions.create(
        model=OPENAI_MODEL, max_tokens=1024, temperature=0, messages=messages,
    )
    track_cost(response, call_type="chat", user=user)
    return response.choices[0].message.content


def respond(message: str, history: list, top_k: int, threshold: float,
            use_web: bool, request: gr.Request):
    user = request.client.host if request and request.client else "unknown"

    yield history, _spinner("Searching corpus…"), "", "", []

    chunks, index, bm25 = _get_artifacts()
    results   = retrieve(message, index, chunks, user=user, bm25=bm25, k=top_k)
    top_score = max((r["score"] for r in results), default=0.0)

    # ── Web search ────────────────────────────────────────────────────────────
    raw_web: list[dict] = []
    web_html = (
        "<p style='color:#8E8E93;font-size:.83em;padding:8px;"
        "font-family:-apple-system,system-ui,sans-serif'>"
        "Web search disabled — enable it in Settings.</p>"
    )
    if use_web:
        yield history, _spinner("Searching the web…"), "", "", []
        raw_web  = _raw_web_results(message)
        web_html = _fetch_web_sources(message)

    yield history, _spinner("Generating answer…"), "", "", []

    # Use web-augmented prompt whenever web is enabled (even if DuckDuckGo returned nothing —
    # the prompt still allows the model to use general knowledge as a fallback)
    if use_web:
        answer = _generate_answer_web_augmented(
            message, results, raw_web, history=history, user=user,
        )
    else:
        answer = generate_answer(message, results, user=user, history=history)

    formatted   = _format_answer(answer, results)
    corpus_html = _build_corpus_html(results, answer, threshold)
    page_imgs   = _render_pdf_pages(results)

    _log_query(message, top_score, len(results))

    new_history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": formatted},
    ]
    yield new_history, "", corpus_html, web_html, page_imgs


# ---------------------------------------------------------------------------
# Image data URIs (base64) — avoids Gradio /file= serving issues in v6
# ---------------------------------------------------------------------------

def _img_data_uri(path: str, max_width: int = 0) -> str:
    """Return a base64 data URI for an image, optionally resizing first."""
    import base64, io
    from PIL import Image
    img = Image.open(path)
    if max_width and img.width > max_width:
        h = int(img.height * max_width / img.width)
        img = img.resize((max_width, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"

_ROOT = Path(__file__).parent.parent
_STATIC = Path(__file__).parent / "static"

_FLAG_PATH    = _ROOT / "european-union-flag-of-europe-flag-of-the-united-states-electrical-switches-others.jpg"
_COLUMNS_PATH = _ROOT / "low-angle-greyscale-shot-ancient-roman-temple-bright-sun.jpg"

def _optional_img(path: Path, **kw) -> str:
    """Decorative image, or an empty string if it is not present.

    These are cosmetic hero images and are not redistributed with the
    repository. A missing decoration must never stop the application from
    starting — previously it raised FileNotFoundError on a fresh clone.
    """
    try:
        return _img_data_uri(str(path), **kw)
    except (FileNotFoundError, OSError):
        return ""


_FLAG_URL    = _optional_img(_FLAG_PATH)                      # decorative only
_COLUMNS_URL = _optional_img(_COLUMNS_PATH, max_width=1600)   # decorative only


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
/* ═══════════════════════════════════════════════════════════════════════════
   Apple.com colour palette — Women's Safety RAG
   ── #FBFBFD page bg   #F5F5F7 section fills   #1D1D1F primary text
   ── #6E6E73 secondary  #0071E3 accent blue     #D2D2D7 borders
   ═══════════════════════════════════════════════════════════════════════════ */

/* ── Override Gradio CSS variables at root — kills dark theme at the source */
:root, .dark {
  --background-fill-primary:    #FFFFFF !important;
  --background-fill-secondary:  #F5F5F7 !important;
  --panel-background-fill:      #FFFFFF !important;
  --panel-border-color:         #D2D2D7 !important;
  --border-color-primary:       #D2D2D7 !important;
  --border-color-accent:        #0071E3 !important;
  --color-accent:               #0071E3 !important;
  --color-accent-soft:          #E8F0FE !important;
  --body-text-color:            #1D1D1F !important;
  --body-text-color-subdued:    #6E6E73 !important;
  --label-text-color:           #6E6E73 !important;
  --input-background-fill:      #FFFFFF !important;
  --input-border-color:         #D2D2D7 !important;
  --stat-background-fill:       #F5F5F7 !important;
  --table-even-background-fill: #FFFFFF !important;
  --table-odd-background-fill:  #F5F5F7 !important;
  --block-background-fill:      #FFFFFF !important;
  --block-label-background-fill:#F5F5F7 !important;
  --block-label-text-color:     #6E6E73 !important;
  --block-label-border-color:   #D2D2D7 !important;
  --block-title-text-color:     #1D1D1F !important;
  --section-header-text-color:  #1D1D1F !important;
  --checkbox-background-color-selected: #0071E3 !important;
  --slider-color:               #0071E3 !important;
  --button-primary-background-fill: #0071E3 !important;
  --button-primary-text-color:  #FFFFFF !important;
  --button-secondary-background-fill: #F5F5F7 !important;
  --button-secondary-text-color: #0071E3 !important;
  --button-secondary-border-color: #D2D2D7 !important;
  --chatbot-code-background-color: #F5F5F7 !important;
}

/* ── Base — containers transparent so body background shows through ──────── */
.gradio-container,
.gradio-container > .wrap,
.gradio-container > .contain,
.gradio-container main,
.gradio-container .main,
#root, #app {
  font-family: -apple-system, "SF Pro Display", "Helvetica Neue",
               "Inter", system-ui, sans-serif !important;
  background: transparent !important;
  color: #1D1D1F !important;
  -webkit-font-smoothing: antialiased;
  position: relative;
  z-index: 1;
}

/* ── Global text — force light theme everywhere ────────────────────────── */
.gradio-container,
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .prose strong,
.gradio-container .prose em {
  color: #1D1D1F !important;
}

/* ── Top-level column/row containers — white cards ──────────────────────── */
.gradio-container > .flex,
.gradio-container .gap-4 > .block:not(.accordion),
.gradio-container .tabs > .tab-content > .block {
  background: #FFFFFF !important;
  border: 1px solid #D2D2D7 !important;
  border-radius: 18px !important;
  box-shadow: 0 2px 10px rgba(0,0,0,.05) !important;
}

/* ── All block backgrounds light ────────────────────────────────────────── */
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel {
  background: #FFFFFF !important;
}

/* ── Block labels / component headers (e.g. "Source pages (click to zoom)") */
.gradio-container .block-label {
  background: #F5F5F7 !important;
  color: #6E6E73 !important;
  border: 1px solid #D2D2D7 !important;
  border-radius: 8px 8px 0 0 !important;
  padding: 5px 12px !important;
  font-size: .73em !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: .06em !important;
}

/* ── Gallery component — light fill, readable label ────────────────────── */
.gradio-container .gallery,
.gradio-container [data-testid="gallery"],
.gradio-container .gallery-container {
  background: #F5F5F7 !important;
  border-radius: 14px !important;
  padding: 10px !important;
}
.gradio-container .gallery .label,
.gradio-container [data-testid="gallery"] > .label,
.gradio-container .gallery .block-label {
  background: #F5F5F7 !important;
  color: #6E6E73 !important;
}
/* Gallery item thumbnails */
.gradio-container .gallery .thumbnail-item,
.gradio-container .gallery figure {
  background: #FFFFFF !important;
  border: 1px solid #D2D2D7 !important;
  border-radius: 10px !important;
}

/* ── Apple-style headings ───────────────────────────────────────────────── */
.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3 {
  color: #1D1D1F !important;
  font-weight: 700 !important;
  letter-spacing: -.02em !important;
  margin-top: 12px !important;
  margin-bottom: 6px !important;
}

/* ── Accordion headers — light pill with dark text ──────────────────────── */
.gradio-container .accordion > .label-wrap {
  background: #F5F5F7 !important;
  border: 1px solid #D2D2D7 !important;
  border-radius: 12px !important;
  padding: 6px 14px !important;
  margin-bottom: 2px !important;
}
.gradio-container .accordion > .label-wrap button,
.gradio-container .accordion > .label-wrap button span,
.gradio-container .accordion > .label-wrap button svg {
  color: #1D1D1F !important;
  fill: #6E6E73 !important;
  font-size: .86em !important;
  font-weight: 600 !important;
}

/* ── Tab navigation — macOS segmented control ───────────────────────────── */
.tab-nav {
  background: #F5F5F7 !important;
  border-radius: 980px !important;
  border: 1px solid #D2D2D7 !important;
  padding: 4px !important;
  display: inline-flex !important;
  gap: 3px !important;
  margin-bottom: 14px !important;
}
.tab-nav button {
  font-size: .84em !important;
  font-weight: 500 !important;
  color: #6E6E73 !important;
  padding: 7px 18px !important;
  border-radius: 980px !important;
  border: none !important;
  background: transparent !important;
  transition: background .15s, color .15s !important;
}
.tab-nav button.selected {
  background: #FFFFFF !important;
  color: #1D1D1F !important;
  font-weight: 700 !important;
  box-shadow: 0 1px 5px rgba(0,0,0,.14) !important;
}

/* ── Text inputs & textareas (NOT checkbox / radio / range) ─────────────── */
.gradio-container input[type="text"],
.gradio-container input[type="search"],
.gradio-container input[type="email"],
.gradio-container input[type="number"],
.gradio-container input:not([type]),
.gradio-container textarea {
  background: #FFFFFF !important;
  border: 1px solid #D2D2D7 !important;
  border-radius: 12px !important;
  color: #1D1D1F !important;
  font-size: .93em !important;
  padding: 10px 14px !important;
}
.gradio-container input[type="text"]:focus,
.gradio-container input[type="search"]:focus,
.gradio-container input[type="number"]:focus,
.gradio-container textarea:focus {
  border-color: #0071E3 !important;
  box-shadow: 0 0 0 3px rgba(0,113,227,.18) !important;
  outline: none !important;
}

/* ── Chat bubbles ───────────────────────────────────────────────────────── */
.message.user {
  background: #E8F0FE !important;
  border-radius: 18px 18px 4px 18px !important;
  color: #1D1D1F !important;
  padding: 12px 16px !important;
}
.message.bot {
  background: #FFFFFF !important;
  border: 1px solid #D2D2D7 !important;
  border-radius: 4px 18px 18px 18px !important;
  color: #1D1D1F !important;
  padding: 12px 16px !important;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
.gradio-container button.primary,
.gradio-container button[variant="primary"] {
  background: #0071E3 !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: 980px !important;
  font-weight: 600 !important;
  font-size: .87em !important;
  padding: 9px 22px !important;
  box-shadow: none !important;
  transition: background .15s !important;
}
.gradio-container button.primary:hover { background: #0077ED !important; }

.gradio-container button.secondary,
.gradio-container button[variant="secondary"] {
  background: #F5F5F7 !important;
  color: #0071E3 !important;
  border: 1px solid #D2D2D7 !important;
  border-radius: 980px !important;
  font-weight: 600 !important;
  font-size: .87em !important;
  padding: 8px 20px !important;
}
.gradio-container button.stop,
.gradio-container button[variant="stop"] {
  background: #FFF0EE !important;
  color: #FF3B30 !important;
  border: 1px solid #FFD5D2 !important;
  border-radius: 980px !important;
  font-weight: 600 !important;
  font-size: .87em !important;
  padding: 8px 20px !important;
}

/* ── Sliders ────────────────────────────────────────────────────────────── */
.gradio-container input[type="range"] {
  accent-color: #0071E3 !important;
}
.gradio-container input[type="range"]::-webkit-slider-thumb {
  background: #0071E3 !important;
}
.gradio-container .wrap > .head > p {
  color: #6E6E73 !important;
  font-size: .78em !important;
}

/* ── Checkboxes & toggles — preserve native size, only recolour ─────────── */
.gradio-container input[type="checkbox"] {
  accent-color: #0071E3 !important;
  width: 16px !important;
  height: 16px !important;
  cursor: pointer !important;
  padding: 0 !important;
  border-radius: 4px !important;
  flex-shrink: 0 !important;
}
/* Gradio wraps checkboxes in a label — make the whole row clickable */
.gradio-container label:has(input[type="checkbox"]) {
  cursor: pointer !important;
  display: flex !important;
  align-items: center !important;
  gap: 8px !important;
  color: #1D1D1F !important;
  font-size: .9em !important;
  padding: 4px 0 !important;
}

/* ── Settings panel ─────────────────────────────────────────────────────── */
.settings-inner {
  background: #F5F5F7 !important;
  border: 1px solid #D2D2D7 !important;
  border-radius: 14px !important;
  padding: 18px 20px !important;
  margin-top: 8px;
}

/* ── Dataframe / table ──────────────────────────────────────────────────── */
.gradio-container table {
  border-radius: 12px !important;
  overflow: hidden !important;
  border: 1px solid #D2D2D7 !important;
}
.gradio-container table th {
  background: #F5F5F7 !important;
  color: #6E6E73 !important;
  font-size: .74em !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: .05em !important;
  padding: 10px 14px !important;
}
.gradio-container table td {
  background: #FFFFFF !important;
  color: #1D1D1F !important;
  font-size: .83em !important;
  border-bottom: 1px solid #F5F5F7 !important;
  padding: 9px 14px !important;
}
.gradio-container table tr:last-child td {
  border-bottom: none !important;
}

/* ── Markdown inside blocks ─────────────────────────────────────────────── */
.gradio-container .prose {
  padding: 6px 2px !important;
}

/* ── File upload ────────────────────────────────────────────────────────── */
.gradio-container .upload-area,
.gradio-container [data-testid="file-upload"] {
  background: #F5F5F7 !important;
  border: 2px dashed #D2D2D7 !important;
  border-radius: 14px !important;
  color: #6E6E73 !important;
  padding: 20px !important;
}

footer { display: none !important; }
"""

# Append background image as a separate f-string so the main CSS stays clean
CSS += f"""
body {{
  background-image: url('{_COLUMNS_URL}') !important;
  background-size: cover !important;
  background-position: center center !important;
  background-attachment: fixed !important;
  background-repeat: no-repeat !important;
}}
body::before {{
  content: '' !important;
  position: fixed !important;
  inset: 0 !important;
  background: rgba(251,251,253,0.87) !important;
  pointer-events: none !important;
  z-index: 0 !important;
}}
"""

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.gray,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("Inter"), "-apple-system", "system-ui"],
)

# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Women's Safety RAG") as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML(f"""
    <div style="
      background: rgba(255,255,255,0.96);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border: 1px solid #D2D2D7;
      border-radius: 20px;
      padding: 22px 28px;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 20px;
      box-shadow: 0 2px 16px rgba(0,0,0,.07);
    ">
      <!-- EU globe flag — top left circle -->
      <img src="{_FLAG_URL}"
           id="eu-flag"
           style="display:block !important;
                  width:72px !important; height:72px !important;
                  min-width:72px !important; min-height:72px !important;
                  max-width:72px !important; max-height:72px !important;
                  border-radius:50% !important;
                  object-fit:cover !important;
                  object-position:center center !important;
                  flex-shrink:0 !important;
                  border:2.5px solid #0071E3 !important;
                  box-shadow:0 3px 12px rgba(0,113,227,.25) !important;
                  background:#E8F0FE !important;"
           onerror="this.outerHTML='<div style=\'width:72px;height:72px;border-radius:50%;background:#0071E3;display:flex;align-items:center;justify-content:center;font-size:2em;flex-shrink:0;border:2.5px solid #0071E3;\'>🇪🇺</div>'" />

      <!-- Title block -->
      <div style="flex:1">
        <div style="
          color:#1D1D1F;
          font-size:1.45em;
          font-weight:700;
          line-height:1.2;
          letter-spacing:-.03em;
          font-family:-apple-system,'SF Pro Display',system-ui,sans-serif;
        ">
          Women's Safety Laws
          <span style="color:#0071E3">&thinsp;EU RAG</span>
        </div>
        <div style="
          color:#6E6E73;
          font-size:.83em;
          margin-top:5px;
          font-family:-apple-system,system-ui,sans-serif;
        ">
          Grounded in EU legal texts, EIGE reports &amp; gender-equality research
        </div>
        <div style="display:flex;gap:7px;margin-top:11px;flex-wrap:wrap">
          <span style="background:#F5F5F7;border:1px solid #D2D2D7;
                       color:#1D1D1F;border-radius:980px;padding:3px 11px;
                       font-size:.73em;font-weight:500">⚖️ Istanbul Convention</span>
          <span style="background:#F5F5F7;border:1px solid #D2D2D7;
                       color:#1D1D1F;border-radius:980px;padding:3px 11px;
                       font-size:.73em;font-weight:500">♀ EIGE Index</span>
          <span style="background:#F5F5F7;border:1px solid #D2D2D7;
                       color:#1D1D1F;border-radius:980px;padding:3px 11px;
                       font-size:.73em;font-weight:500">🛡️ GBV Directive</span>
        </div>
      </div>
    </div>
    """)

    stats_html = gr.HTML(
        "<div style='height:48px;background:#F5F5F7;border:1px solid #D2D2D7;border-radius:14px;"
        "display:flex;align-items:center;padding:0 16px;color:#6E6E73;"
        "font-size:.8em;font-weight:500;"
        "font-family:-apple-system,system-ui,sans-serif'>"
        "Corpus stats load after first query…</div>"
    )

    with gr.Row(equal_height=False):

        # ── Left: chat ───────────────────────────────────────────────────────
        with gr.Column(scale=3, min_width=400):

            chatbot = gr.Chatbot(
                height=440,
                show_label=False,
                avatar_images=(None, None),
                placeholder=(
                    "<div style='text-align:center;padding:40px 20px;"
                    "font-family:-apple-system,system-ui,sans-serif'>"
                    "<div style='font-size:2.6em;margin-bottom:10px'>⚖️</div>"
                    "<div style='font-size:1.05em;font-weight:700;color:#1D1D1F;"
                    "margin-bottom:6px;letter-spacing:-.02em'>EU Women's Safety Q&amp;A</div>"
                    "<div style='font-size:.84em;color:#6E6E73;line-height:1.6'>"
                    "Istanbul Convention · GREVIO · EIGE Gender Index<br>"
                    "Gender-Based Violence Directive · Equal Pay · Work-Life Balance"
                    "</div></div>"
                ),
            )

            status_html = gr.HTML("", elem_classes="status-wrap")

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask a question about EU women's safety laws…",
                    show_label=False, scale=5, autofocus=True,
                    max_lines=4, container=False,
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1, min_width=70)

            with gr.Row(elem_classes="toolbar"):
                clear_btn  = gr.Button("Clear chat",    size="sm", variant="secondary", scale=1)
                export_btn = gr.Button("Export chat",   size="sm", variant="secondary", scale=1)

            export_file = gr.File(label="Download", visible=False, interactive=False)

            # Settings
            with gr.Accordion("Settings", open=False):
                top_k_slider = gr.Slider(
                    minimum=3, maximum=15, value=8, step=1,
                    label="Retrieved passages (top-k)",
                    info="More = richer context, slightly slower",
                )
                threshold_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.50, step=0.05,
                    label="Confidence threshold",
                    info="Scores below this trigger a low-confidence warning",
                )
                web_search_toggle = gr.Checkbox(
                    value=True,
                    label="Enable live web sources",
                    info="Appends top DuckDuckGo results to the Sources panel",
                )

            # HITL
            with gr.Accordion("Human Review (HITL)", open=False):
                gr.Markdown(
                    "Review the last answer — approve, reject, or request a rewrite.",
                    elem_classes="src-panel",
                )
                with gr.Row():
                    approve_btn = gr.Button("✓ Approve & save", variant="primary", size="sm")
                    reject_btn  = gr.Button("✗ Reject",         variant="stop",    size="sm")
                hitl_status   = gr.Markdown("")
                rewrite_input = gr.Textbox(
                    placeholder="Describe what needs to change…",
                    label="Rewrite feedback", lines=2, container=True,
                )
                rewrite_btn = gr.Button("Request rewrite", variant="secondary", size="sm")

        # ── Right: tabbed panel ──────────────────────────────────────────────
        with gr.Column(scale=2, min_width=300):

            with gr.Tabs():

                # ── Sources tab ──────────────────────────────────────────────
                with gr.TabItem("Sources"):
                    sources_gallery = gr.Gallery(
                        label="Source pages (click to zoom)",
                        show_label=True, columns=2,
                        object_fit="contain", height=300, preview=True,
                    )
                    with gr.Accordion("Corpus excerpts", open=True):
                        corpus_html_display = gr.HTML(
                            "<p style='color:#8E8E93;font-size:.83em;padding:8px;"
                            "font-family:-apple-system,system-ui,sans-serif'>"
                            "Sources appear here after your first question.</p>"
                        )
                    with gr.Accordion("Web sources", open=True):
                        web_html_display = gr.HTML(
                            "<p style='color:#8E8E93;font-size:.83em;padding:8px;"
                            "font-family:-apple-system,system-ui,sans-serif'>"
                            "Web sources appear here after your first question.</p>"
                        )

                # ── Knowledge Base tab ───────────────────────────────────────
                with gr.TabItem("Knowledge Base"):
                    gr.Markdown("**Add text to the live corpus:**")
                    add_text_input  = gr.Textbox(
                        placeholder="Paste a fact, paragraph, or policy update…",
                        label="Text to add", lines=4,
                    )
                    add_text_btn    = gr.Button("Add to corpus", variant="primary", size="sm")
                    add_text_status = gr.Markdown("")

                    gr.Markdown("**Upload a PDF document:**")
                    pdf_upload        = gr.File(label="PDF file", file_types=[".pdf"])
                    ingest_pdf_btn    = gr.Button("Ingest PDF", variant="primary", size="sm")
                    ingest_pdf_status = gr.Markdown("")

                # ── Analytics tab ────────────────────────────────────────────
                with gr.TabItem("Analytics"):
                    analytics_summary = gr.HTML(_analytics_summary_html([], []))
                    refresh_analytics_btn = gr.Button(
                        "Refresh analytics", size="sm", variant="secondary",
                    )
                    plot_questions  = gr.Plot(label="Top Questions")
                    plot_daily      = gr.Plot(label="Daily Usage")
                    plot_feedback   = gr.Plot(label="Feedback")
                    plot_confidence = gr.Plot(label="Confidence Scores")
                    gr.Markdown("**Recent queries**")
                    recent_queries_table = gr.Dataframe(
                        headers=["time", "query", "score"],
                        datatype=["str", "str", "number"],
                        interactive=False,
                        row_count=(10, "fixed"),
                        wrap=True,
                    )

                # ── Feedback log tab ─────────────────────────────────────────
                with gr.TabItem("Feedback Log"):
                    feedback_display = gr.HTML(_render_feedback_html([]))
                    refresh_feedback_btn = gr.Button("Refresh", size="sm", variant="secondary")

    # ── Wire events ──────────────────────────────────────────────────────────

    submit_inputs  = [msg_input, chatbot, top_k_slider, threshold_slider, web_search_toggle]
    submit_outputs = [chatbot, status_html, corpus_html_display, web_html_display, sources_gallery]

    msg_input.submit(respond, inputs=submit_inputs, outputs=submit_outputs).then(
        lambda: "", outputs=msg_input,
    ).then(_stats_html, outputs=stats_html)

    submit_btn.click(respond, inputs=submit_inputs, outputs=submit_outputs).then(
        lambda: "", outputs=msg_input,
    ).then(_stats_html, outputs=stats_html)

    clear_btn.click(
        lambda: ([], "", "", "", []),
        outputs=[chatbot, status_html, corpus_html_display, web_html_display, sources_gallery],
    )

    chatbot.like(handle_like, inputs=[chatbot], outputs=None)
    export_btn.click(export_conversation, inputs=[chatbot], outputs=[export_file])

    # HITL
    approve_btn.click(approve_last_answer,  inputs=[chatbot], outputs=[hitl_status])
    reject_btn.click(reject_last_answer,    inputs=[chatbot], outputs=[chatbot, hitl_status])
    rewrite_btn.click(
        request_rewrite, inputs=[rewrite_input, chatbot], outputs=[chatbot, hitl_status],
    ).then(lambda: "", outputs=rewrite_input)

    # Knowledge base
    add_text_btn.click(
        handle_add_text, inputs=[add_text_input], outputs=[add_text_status, stats_html],
    )
    ingest_pdf_btn.click(
        handle_ingest_pdf, inputs=[pdf_upload], outputs=[ingest_pdf_status, stats_html],
    )

    # Analytics — only runs when user clicks Refresh (no auto-load to keep startup fast)
    refresh_analytics_btn.click(
        refresh_analytics,
        outputs=[analytics_summary, plot_questions, plot_daily,
                 plot_feedback, plot_confidence, recent_queries_table],
    )

    # Feedback log
    refresh_feedback_btn.click(
        lambda: _render_feedback_html(_load_feedback()[-10:][::-1]),
        outputs=[feedback_display],
    )


if __name__ == "__main__":
    demo.launch(
        css=CSS,
        theme=_THEME,
        allowed_paths=[
            tempfile.gettempdir(),
            str(PDF_INPUT_DIR.resolve()),
            str(_STATIC.resolve()),
        ],
    )
