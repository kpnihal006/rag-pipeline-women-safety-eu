from __future__ import annotations

"""
app/mcp_server.py

FastMCP server exposing RAG-augmentation tools:
  - web_search             — DuckDuckGo live search
  - scrape_url             — BeautifulSoup page scraper (SSRF-guarded)
  - create_markdown_report — persist a markdown report to disk
  - add_to_database        — embed text and add it to the live FAISS index
  - ingest_pdf             — extract, chunk, embed, and index a new PDF
  - generate_chart         — matplotlib chart generator
  - create_mermaid_diagram — mermaid diagram generator (.mmd + .md)
  - security_report        — what the guardrails caught this session

Every tool that touches the outside world (web, filesystem, vector store) is
wrapped by `app.security.guard` and traced by `app.observability.tracer`.

Run standalone:
    uv run python app/mcp_server.py
"""

import json
import logging
import os
import re
import sys
import threading
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import faiss
import numpy as np
import requests as http_requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.chunk import (
    OPENAI_EMBED_MODEL,
    _get_openai_client,
    _split,
    load_artifacts,
    save_artifacts,
)
from scripts.extract import clean_text
from app.observability import tracer
from app.security import SecurityError, guard

load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

mcp = FastMCP("women-safety-rag")

# ---------------------------------------------------------------------------
# Shared state — lazy-loaded FAISS artifacts (thread-safe writes)
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.environ.get("PDF_OUTPUT_DIR", "data"))
REPORTS_DIR = Path("reports")
CHARTS_DIR = REPORTS_DIR / "charts"

_lock: threading.Lock = threading.Lock()
_chunks: list[dict] = []
_index: faiss.Index | None = None
_bm25 = None
_artifacts_loaded = False


def _ensure_loaded() -> None:
    """Lazy-load FAISS artifacts on first tool call."""
    global _chunks, _index, _bm25, _artifacts_loaded
    if _artifacts_loaded:
        return
    with _lock:
        if _artifacts_loaded:
            return
        try:
            _chunks, _index, _bm25 = load_artifacts(DATA_DIR)
            _artifacts_loaded = True
            log.info("MCP server: artifacts loaded (%d chunks)", len(_chunks))
        except Exception as exc:
            log.warning("MCP server: could not load artifacts: %s", exc)


# ---------------------------------------------------------------------------
# Tool: web_search
# ---------------------------------------------------------------------------

@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return a JSON list of results.

    Each result has: title, href, body.

    Result text is untrusted: search snippets are attacker-controllable, so
    every title/body is passed through the prompt-injection scanner before it
    is returned to the model.
    """
    import time
    import warnings

    with tracer.span("tool", "web_search", query=query[:200]) as sp:
        guard.check_rate("web_search", limit=60, window_s=60.0)
        max_results = max(1, min(int(max_results), 15))

        last_exc: Exception | None = None
        results: list[dict] = []
        for attempt in range(3):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    results = list(DDGS().text(query, max_results=max_results))
                if results:
                    break
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))

        if not results:
            sp.attributes["error"] = str(last_exc)
            return json.dumps({"error": str(last_exc), "query": query})

        flagged = 0
        sanitised: list[dict] = []
        for r in results:
            host = urlparse(r.get("href", "")).netloc or "web"
            entry = dict(r)
            for field_name in ("title", "body"):
                if entry.get(field_name):
                    verdict = guard.scan_untrusted(
                        str(entry[field_name]), origin=f"web:{host}"
                    )
                    entry[field_name] = verdict.sanitised
                    flagged += len(verdict.events)
            sanitised.append(entry)

        sp.attributes["n_results"] = len(sanitised)
        sp.attributes["security_events"] = flagged
        if flagged:
            tracer.event("prompt-injection patterns neutralised in search results",
                         count=flagged)

        return json.dumps(sanitised, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Tool: scrape_url
# ---------------------------------------------------------------------------

@mcp.tool()
def scrape_url(url: str, max_chars: int = 4000) -> str:
    """Fetch a URL and return its cleaned text content (BeautifulSoup).

    Strips scripts, styles, nav, header, and footer elements.

    Guarded: the URL is checked for SSRF (private ranges, cloud metadata
    endpoints, non-HTTP schemes) before the request, and the page body is
    scanned for prompt-injection payloads before it is returned. Scraped page
    text is the single most attacker-controllable input in the system.
    """
    with tracer.span("tool", "scrape_url", url=url[:200]) as sp:
        try:
            guard.check_rate("scrape_url", limit=40, window_s=60.0)
            guard.check_url(url)
        except SecurityError as exc:
            sp.attributes["blocked"] = True
            return f"[scrape_url blocked by security guard] {exc}"

        try:
            resp = http_requests.get(
                url,
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0"},
                allow_redirects=True,
            )
            resp.raise_for_status()

            # A redirect can land somewhere the pre-flight check never saw.
            if resp.url != url:
                try:
                    guard.check_url(str(resp.url))
                except SecurityError as exc:
                    sp.attributes["blocked"] = True
                    return f"[scrape_url blocked after redirect] {exc}"

            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            cleaned = "\n".join(lines)
            cleaned = cleaned[:max_chars] + ("…" if len(cleaned) > max_chars else "")

            host = urlparse(url).netloc or "web"
            verdict = guard.scan_untrusted(cleaned, origin=f"web:{host}")
            sp.attributes["chars"] = len(verdict.sanitised)
            sp.attributes["security_events"] = len(verdict.events)
            if verdict.events:
                tracer.event("page content flagged",
                             rules=",".join(e.rule_id for e in verdict.events))
            return verdict.sanitised
        except Exception as exc:
            sp.attributes["error"] = str(exc)
            return f"[scrape_url error] {exc}"


# ---------------------------------------------------------------------------
# Tool: create_markdown_report
# ---------------------------------------------------------------------------

@mcp.tool()
def create_markdown_report(
    title: str,
    content: str,
    filename: str | None = None,
) -> str:
    """Save a markdown report to reports/ and return the file path.

    Guarded: the filename is stripped of path separators and the resolved
    target must stay inside reports/, so a model that has swallowed an
    injected "save this to ~/.ssh/authorized_keys" cannot act on it. The body
    is redacted for secrets before it is written to disk.
    """
    with tracer.span("tool", "create_markdown_report", title=title[:120]) as sp:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        if not filename:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in title[:40])
            filename = f"{ts}_{safe}.md"
        filename = guard.safe_filename(filename, default="report")
        if not filename.endswith(".md"):
            filename += ".md"

        redaction = guard.redact(content, origin="report_body")
        title_redaction = guard.redact(title, origin="report_title")
        sp.attributes["redactions"] = len(redaction.events) + len(title_redaction.events)

        try:
            report_path = guard.check_write_path(REPORTS_DIR / filename)
        except SecurityError as exc:
            sp.attributes["blocked"] = True
            return f"[create_markdown_report blocked by security guard] {exc}"

        full_content = f"# {title_redaction.sanitised}\n\n{redaction.sanitised}\n"
        report_path.write_text(full_content, encoding="utf-8")
        sp.attributes["bytes"] = len(full_content)
        return str(report_path.relative_to(Path.cwd().resolve()))


# ---------------------------------------------------------------------------
# Tool: add_to_database
# ---------------------------------------------------------------------------

@mcp.tool()
def add_to_database(
    text: str,
    source: str = "dynamic",
    page: int = 1,
) -> str:
    """Split text into chunks, embed them, and add to the live FAISS index.

    The updated index and chunk list are persisted to disk immediately.
    Returns a summary of how many chunks were added.
    """
    with tracer.span("tool", "add_to_database", source=source) as sp:
        _ensure_loaded()
        if _index is None:
            return "[add_to_database] FAISS index not loaded — run chunk.py first."

        # Text written into the vector store is a *persistent* injection vector:
        # a payload stored today is retrieved into every future prompt. Scan and
        # neutralise before it is ever embedded.
        verdict = guard.scan_untrusted(text, origin=f"db_write:{source}")
        sp.attributes["security_events"] = len(verdict.events)
        if verdict.blocked:
            sp.attributes["blocked"] = True
            rules = ", ".join(e.rule_id for e in verdict.events)
            return (
                "[add_to_database] Refused: the submitted text contains "
                f"instruction-injection patterns ({rules}). Nothing was stored."
            )

        clean = guard.redact(verdict.sanitised, origin=f"db_write:{source}").sanitised
        source = guard.safe_filename(source, default="dynamic")
        page = max(1, int(page))

        new_chunks_text = _split(clean)
        if not new_chunks_text:
            return "[add_to_database] No chunks produced from the provided text."

        client = _get_openai_client()
        try:
            response = client.embeddings.create(
                model=OPENAI_EMBED_MODEL,
                input=new_chunks_text,
                encoding_format="float",
                dimensions=_index.d,
            )
        except Exception as exc:
            return f"[add_to_database] Embedding failed: {exc}"

        vectors = np.array([d.embedding for d in response.data], dtype="float32")
        faiss.normalize_L2(vectors)

        new_chunk_dicts: list[dict] = []
        for i, chunk_text in enumerate(new_chunks_text):
            chunk_id = f"{source}::p{page}::dyn{i}"
            new_chunk_dicts.append({
                "chunk_id": chunk_id,
                "source": source,
                "page": page,
                "chunk_index": i,
                "extractor": "dynamic",
                "text": chunk_text,
            })

        with _lock:
            _index.add(vectors)
            _chunks.extend(new_chunk_dicts)
            save_artifacts(_chunks, _index, DATA_DIR)

        sp.attributes["chunks_added"] = len(new_chunk_dicts)
        sp.attributes["index_size"] = _index.ntotal
        return (
            f"Added {len(new_chunk_dicts)} chunk(s) from '{source}' (page {page}) "
            f"to the database. Index now has {_index.ntotal} vectors."
        )


# ---------------------------------------------------------------------------
# Tool: ingest_pdf
# ---------------------------------------------------------------------------

@mcp.tool()
def ingest_pdf(pdf_path: str) -> str:
    """Extract text from a PDF, chunk it, embed it, and add it to the FAISS index.

    The PDF path must be accessible from the server's working directory.
    Returns a summary of pages and chunks ingested.
    """
    _ensure_loaded()
    if _index is None:
        return "[ingest_pdf] FAISS index not loaded — run chunk.py first."

    try:
        path = guard.check_read_path(pdf_path)
    except SecurityError as exc:
        return f"[ingest_pdf blocked by security guard] {exc}"

    if path.suffix.lower() != ".pdf":
        return f"[ingest_pdf] Not a PDF file: {path.name}"

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        return f"[ingest_pdf] Could not open PDF: {exc}"

    # Extract raw pages
    raw_pages: list[dict] = []
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            raw_pages.append({
                "source": path.name,
                "page": page_num + 1,
                "text": text,
            })
    finally:
        doc.close()

    if not raw_pages:
        return "[ingest_pdf] No pages extracted."

    # Clean and chunk. A user-supplied PDF is untrusted input — white-on-white
    # text carrying instructions is a documented attack on document-ingesting
    # agents — so every page is scanned before it reaches the index.
    flagged_pages = 0
    all_new_chunks: list[dict] = []
    all_texts: list[str] = []
    for entry in raw_pages:
        cleaned = clean_text(entry["text"])
        page_verdict = guard.scan_untrusted(
            cleaned, origin=f"pdf:{path.name}#p{entry['page']}"
        )
        if page_verdict.events:
            flagged_pages += 1
        cleaned = page_verdict.sanitised
        for i, chunk_text in enumerate(_split(cleaned)):
            chunk_id = f"{path.name}::p{entry['page']}::c{i}"
            all_new_chunks.append({
                "chunk_id": chunk_id,
                "source": path.name,
                "page": entry["page"],
                "chunk_index": i,
                "extractor": "ingest_pdf",
                "text": chunk_text,
            })
            all_texts.append(chunk_text)

    if not all_texts:
        return "[ingest_pdf] No chunks produced after cleaning."

    # Embed in batches of 512
    client = _get_openai_client()
    vectors_list: list[list[float]] = []
    batch_size = 512
    try:
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i : i + batch_size]
            response = client.embeddings.create(
                model=OPENAI_EMBED_MODEL,
                input=batch,
                encoding_format="float",
                dimensions=_index.d,
            )
            vectors_list.extend([d.embedding for d in response.data])
    except Exception as exc:
        return f"[ingest_pdf] Embedding failed: {exc}"

    vectors = np.array(vectors_list, dtype="float32")
    faiss.normalize_L2(vectors)

    with _lock:
        _index.add(vectors)
        _chunks.extend(all_new_chunks)
        save_artifacts(_chunks, _index, DATA_DIR)

    note = (
        f" {flagged_pages} page(s) contained instruction-like text and were "
        "neutralised before indexing." if flagged_pages else ""
    )
    return (
        f"Ingested '{path.name}': {len(raw_pages)} pages → "
        f"{len(all_new_chunks)} chunks added. "
        f"Index now has {_index.ntotal} vectors.{note}"
    )


# ---------------------------------------------------------------------------
# Tool: generate_chart
# ---------------------------------------------------------------------------

@mcp.tool()
def generate_chart(
    chart_type: str,
    data: str,
    title: str,
    x_label: str = "",
    y_label: str = "",
) -> str:
    """Generate a chart using matplotlib and save it to reports/charts/.

    Args:
        chart_type: One of "bar", "line", "pie", "scatter".
        data: JSON string. For bar/line/scatter: {"labels": [...], "values": [...]}.
              For pie: {"labels": [...], "sizes": [...]}.
              For scatter: {"x": [...], "y": [...]}.
        title: Chart title (also used as filename base).
        x_label: X-axis label (bar, line, scatter).
        y_label: Y-axis label (bar, line, scatter).

    Returns:
        File path to the saved chart image.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        return f"[generate_chart] Invalid JSON data: {exc}"

    fig, ax = plt.subplots(figsize=(10, 6))

    try:
        chart_type = chart_type.lower().strip()
        if chart_type == "bar":
            labels = payload["labels"]
            values = payload["values"]
            ax.bar(labels, values)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            plt.xticks(rotation=45, ha="right")
        elif chart_type == "line":
            labels = payload["labels"]
            values = payload["values"]
            ax.plot(labels, values, marker="o")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            plt.xticks(rotation=45, ha="right")
        elif chart_type == "pie":
            labels = payload["labels"]
            sizes = payload["sizes"]
            ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
            ax.axis("equal")
        elif chart_type == "scatter":
            x = payload["x"]
            y = payload["y"]
            ax.scatter(x, y)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
        else:
            plt.close(fig)
            return f"[generate_chart] Unsupported chart type: {chart_type}"
    except (KeyError, TypeError) as exc:
        plt.close(fig)
        return f"[generate_chart] Data format error: {exc}"

    ax.set_title(title)
    plt.tight_layout()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in title[:40])
    out_path = CHARTS_DIR / f"{ts}_{safe}.png"
    try:
        out_path = guard.check_write_path(out_path)
    except SecurityError as exc:
        plt.close(fig)
        return f"[generate_chart blocked by security guard] {exc}"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(out_path.relative_to(Path.cwd().resolve()))


# ---------------------------------------------------------------------------
# Tool: create_mermaid_diagram
# ---------------------------------------------------------------------------

_MERMAID_KINDS = {
    "flowchart", "graph", "sequenceDiagram", "classDiagram", "stateDiagram-v2",
    "erDiagram", "journey", "gantt", "pie", "mindmap", "timeline", "quadrantChart",
}


@mcp.tool()
def create_mermaid_diagram(
    title: str,
    diagram: str,
    diagram_type: str = "flowchart",
) -> str:
    """Save a Mermaid diagram to reports/diagrams/ as both .mmd and .md.

    Use this for structure that a bar chart cannot express: the route a query
    takes through the agent team, the relationship between legal instruments,
    a timeline of policy milestones, or a state machine of the review loop.

    Args:
        title: Diagram title (also the filename base).
        diagram: Mermaid source. May include or omit the leading type line —
            if omitted, `diagram_type` is prepended.
        diagram_type: One of flowchart, graph, sequenceDiagram, classDiagram,
            stateDiagram-v2, erDiagram, journey, gantt, pie, mindmap, timeline,
            quadrantChart.

    Returns:
        Path to the saved .md file (renders directly on GitHub).
    """
    with tracer.span("tool", "create_mermaid_diagram", title=title[:120]) as sp:
        diagrams_dir = REPORTS_DIR / "diagrams"
        diagrams_dir.mkdir(parents=True, exist_ok=True)

        body = (diagram or "").strip()
        if not body:
            return "[create_mermaid_diagram] Empty diagram source."

        # Mermaid source is code the browser will execute on render — strip any
        # HTML/script the model may have wrapped around it.
        if "<script" in body.lower() or "</" in body.lower():
            body = re.sub(r"<[^>]+>", "", body)
            sp.attributes["stripped_html"] = True

        first_word = body.split(None, 1)[0] if body.split() else ""
        if first_word not in _MERMAID_KINDS:
            kind = diagram_type.strip()
            if kind not in _MERMAID_KINDS:
                return (
                    f"[create_mermaid_diagram] Unsupported diagram_type: "
                    f"{diagram_type!r}. Choose one of: {', '.join(sorted(_MERMAID_KINDS))}"
                )
            body = f"{kind}\n{body}"

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = guard.safe_filename(
            "".join(c if c.isalnum() or c in "-_" else "_" for c in title[:40]),
            default="diagram",
        )
        stem = f"{ts}_{safe}"

        try:
            mmd_path = guard.check_write_path(diagrams_dir / f"{stem}.mmd")
            md_path = guard.check_write_path(diagrams_dir / f"{stem}.md")
        except SecurityError as exc:
            sp.attributes["blocked"] = True
            return f"[create_mermaid_diagram blocked by security guard] {exc}"

        mmd_path.write_text(body + "\n", encoding="utf-8")
        md_path.write_text(
            f"# {title}\n\n```mermaid\n{body}\n```\n", encoding="utf-8"
        )

        sp.attributes["lines"] = body.count("\n") + 1
        cwd = Path.cwd().resolve()
        return (
            f"Diagram saved → {md_path.relative_to(cwd)} "
            f"(source: {mmd_path.relative_to(cwd)})"
        )


# ---------------------------------------------------------------------------
# Tool: security_report
# ---------------------------------------------------------------------------

@mcp.tool()
def security_report() -> str:
    """Return everything the security guardrails caught this session, as JSON.

    Exposes prompt-injection hits, redactions, blocked URLs, and refused file
    paths so a reviewer can audit what the pipeline defended against.
    """
    return json.dumps(guard.report(), indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        stream=sys.stdout,
    )
    mcp.run()
