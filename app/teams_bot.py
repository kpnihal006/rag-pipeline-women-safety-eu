from __future__ import annotations

"""
app/teams_bot.py

Microsoft Teams Outgoing Webhook with:
  - Plain text responses with source citations
  - Per-conversation settings (web search, top-k, threshold)
  - Web search fallback when corpus confidence is low

Commands:
  /ask <q>         — answer from corpus (+ web fallback)
  /quick <q>       — same as /ask
  /settings        — show current session settings
  /web on|off      — toggle live web search
  /topk N          — set retrieved passages (3-15)
  /threshold N     — set confidence threshold (0.1-0.9)
  /add <text>      — add text to knowledge base
  /pdf <path>      — ingest PDF into knowledge base
  /stats           — corpus statistics
  /reset           — clear conversation history
  /help            — this message

Setup:
  1. uv run python app/teams_bot.py
  2. ngrok http 3978   (in a second terminal)
  3. Teams → Manage team → Apps → Create outgoing webhook
     Callback URL: https://<ngrok-url>/webhook
  4. Copy security token → .env: TEAMS_SECURITY_TOKEN=<token>

.env variables:
    TEAMS_SECURITY_TOKEN  — HMAC token from Teams (required for auth)
    TEAMS_BOT_PORT        — port (default 3978)
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

from aiohttp import web
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.mcp_server import add_to_database, ingest_pdf

load_dotenv()

log = logging.getLogger(__name__)

SECURITY_TOKEN = os.environ.get("TEAMS_SECURITY_TOKEN", "")
PORT           = int(os.environ.get("TEAMS_BOT_PORT", "3978"))
MAX_LEN        = 4000

# ---------------------------------------------------------------------------
# Per-conversation state
# ---------------------------------------------------------------------------

_history:  dict[str, list[dict]] = defaultdict(list)
_settings: dict[str, dict]       = defaultdict(lambda: {
    "web":       True,
    "topk":      8,
    "threshold": 0.40,
})
MAX_TURNS = 10


# ---------------------------------------------------------------------------
# Artifact cache — loaded once at startup
# ---------------------------------------------------------------------------

_chunks: list = []
_index        = None
_bm25         = None
_artifacts_ready = False


def _get_artifacts():
    global _chunks, _index, _bm25, _artifacts_ready
    if not _artifacts_ready:
        from scripts.chunk import load_artifacts
        data_dir = Path(os.environ.get("PDF_OUTPUT_DIR", "data"))
        _chunks, _index, _bm25 = load_artifacts(data_dir)
        _artifacts_ready = True
        log.info("Artifacts loaded: %d chunks", len(_chunks))
    return _chunks, _index, _bm25


# ---------------------------------------------------------------------------
# HMAC verification
# ---------------------------------------------------------------------------

def _verify_hmac(body: bytes, auth_header: str) -> bool:
    if not SECURITY_TOKEN:
        log.warning("TEAMS_SECURITY_TOKEN not set — skipping auth check.")
        return True
    if not auth_header.startswith("HMAC "):
        log.warning("No HMAC header — got: %r", auth_header[:60])
        return False
    try:
        received_b64 = auth_header[5:].strip()
        key          = base64.b64decode(SECURITY_TOKEN.strip())
        expected_b64 = base64.b64encode(
            hmac.new(key, body, hashlib.sha256).digest()
        ).decode()
        match = hmac.compare_digest(received_b64, expected_b64)
        if not match:
            log.warning("HMAC mismatch — received: %s  expected: %s",
                        received_b64[:20], expected_b64[:20])
        return match
    except Exception as exc:
        log.warning("HMAC error: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str) -> str:
    return text if len(text) <= MAX_LEN else text[:MAX_LEN - 3] + "..."


def _strip_mention(text: str) -> str:
    text = re.sub(r"<at>[^<]*</at>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return (text
            .replace("&nbsp;", " ").replace("&amp;", "&")
            .replace("&lt;", "<").replace("&gt;", ">")
            .replace("&quot;", '"')).strip()


def _add_history(conv_id: str, role: str, content: str) -> None:
    hist = _history[conv_id]
    hist.append({"role": role, "content": content})
    if len(hist) > MAX_TURNS * 2:
        _history[conv_id] = hist[-(MAX_TURNS * 2):]


def _get_history(conv_id: str) -> list[dict]:
    return list(_history[conv_id])


def _teams_reply(text: str) -> web.Response:
    return web.json_response({"type": "message", "text": _truncate(text)})


# ---------------------------------------------------------------------------
# Web-augmented LLM helper (sync — called via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _web_augmented_answer(
    query: str,
    corpus_results: list[dict],
    web_results: list[dict],
    history: list,
) -> str:
    from scripts.chunk import OPENAI_MODEL, _get_openai_client, track_cost

    corpus_ctx = "\n\n---\n\n".join(
        f"[CORPUS: {r['source']} p.{r['page']}]\n{r['text'][:600]}"
        for r in corpus_results[:5]
    ) or "(no corpus excerpts)"
    web_ctx = "\n\n---\n\n".join(
        f"[WEB: {r.get('title','')[:60]}]\n{r.get('body','')[:400]}"
        for r in web_results if r.get("body")
    ) or "(no web results)"

    system = (
        "You are a legal research assistant for EU women's safety laws.\n"
        "Use CORPUS EXCERPTS first, then WEB SNIPPETS, then your training knowledge.\n"
        "Be concise — answer in 3–5 sentences. Cite sources inline.\n\n"
        f"CORPUS:\n{corpus_ctx}\n\nWEB:\n{web_ctx}"
    )
    msgs: list[dict] = [{"role": "system", "content": system}]
    for turn in (history or [])[-6:]:
        if isinstance(turn, dict):
            msgs.append({"role": turn["role"], "content": turn["content"]})
    msgs.append({"role": "user", "content": query})

    client = _get_openai_client()
    resp   = client.chat.completions.create(
        model=OPENAI_MODEL, max_tokens=500, temperature=0, messages=msgs,
    )
    track_cost(resp, call_type="chat")
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Core answer function
# ---------------------------------------------------------------------------

async def _answer(query: str, conv_id: str) -> tuple[str, list[str]]:
    """Return (answer_text, source_labels).

    Optimised for Teams' 5-second webhook timeout:
    - Query expansion (LLM call) is skipped to save ~1 s.
    - Retrieval and web search run concurrently via asyncio.
    - max_tokens capped at 500 for faster generation.
    - All blocking I/O offloaded to threads so the event loop stays free.
    """
    import asyncio
    from scripts.chunk import retrieve, generate_answer, OPENAI_MODEL, _get_openai_client, track_cost

    cfg       = _settings[conv_id]
    top_k     = cfg["topk"]
    threshold = cfg["threshold"]
    use_web   = cfg["web"]

    chunks, index, bm25 = _get_artifacts()
    history = _get_history(conv_id)

    # ── Retrieval (rerank=True so scores are cross-encoder logits comparable
    #    to the 0.50 threshold; expand=False still skips query expansion to
    #    save ~1 s of latency) ────────────────────────────────────────────────
    results = await asyncio.to_thread(
        retrieve, query, index, chunks,
        bm25=bm25, k=top_k, rerank=True, expand=True,
    )
    top_score = max((r["score"] for r in results), default=0.0)

    # ── Web search (parallel, only when low confidence) ────────────────────
    async def _web_search() -> list[dict]:
        if not use_web:
            return []
        def _sync():
            try:
                import warnings
                from duckduckgo_search import DDGS
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return list(DDGS().text(
                        query + " EU law women rights", max_results=3))
            except Exception:
                return []
        return await asyncio.to_thread(_sync)

    # ── Web search: start immediately in background whenever web is enabled ──
    # When use_web=True, we always include web results regardless of confidence.
    # When use_web=False, fall back to web only on low confidence.
    web_task = asyncio.create_task(_web_search()) if use_web else (
        asyncio.create_task(_web_search()) if top_score < threshold else None
    )

    # ── Answer generation ─────────────────────────────────────────────────
    high_confidence = top_score >= threshold

    if use_web:
        # Always web-augmented when web is toggled on
        web_results: list[dict] = await web_task
        answer = await asyncio.to_thread(
            _web_augmented_answer, query, results, web_results, history
        )
    elif high_confidence:
        answer = await asyncio.to_thread(generate_answer, query, results, history=history)
    else:
        # Low confidence, web off → web-augmented with empty web results
        log.info("Low-confidence retrieval: top score %.3f < threshold %.2f for query: %s…",
                 top_score, threshold, query[:60])
        web_results = await web_task if web_task else []
        answer = await asyncio.to_thread(
            _web_augmented_answer, query, results, web_results, history
        )

    # ── Source labels ──────────────────────────────────────────────────────
    sources: list[str] = []
    seen: set = set()
    for r in results:
        key = (r["source"], r["page"])
        if key not in seen:
            seen.add(key)
            sources.append(f"{r['source']} p.{r['page']}")

    return _truncate(answer), sources


# ---------------------------------------------------------------------------
# Settings command handler
# ---------------------------------------------------------------------------

def _handle_settings_cmd(message: str, lower: str, conv_id: str) -> web.Response | None:
    cfg = _settings[conv_id]

    # /settings — show current
    if lower in ("/settings", "settings"):
        return _teams_reply(
            f"⚙️ **Current settings for this conversation**\n\n"
            f"- 🌐 Web search: **{'on' if cfg['web'] else 'off'}**\n"
            f"- 🔢 Top-k passages: **{cfg['topk']}**\n"
            f"- 📊 Confidence threshold: **{cfg['threshold']}**\n\n"
            "_Change with /web on|off  · /topk N  · /threshold N_"
        )

    # /web on|off
    if lower in ("/web on", "web on"):
        cfg["web"] = True
        return _teams_reply("🌐 Web search **enabled** for this conversation.")
    if lower in ("/web off", "web off"):
        cfg["web"] = False
        return _teams_reply("🌐 Web search **disabled** for this conversation.")

    # /topk N
    if lower.startswith("/topk ") or lower.startswith("topk "):
        try:
            n = int(message.split()[-1])
            if 3 <= n <= 15:
                cfg["topk"] = n
                return _teams_reply(f"🔢 Top-k set to **{n}**.")
            return _teams_reply("Top-k must be between 3 and 15.")
        except ValueError:
            return _teams_reply("Usage: /topk 5")

    # /threshold N
    if lower.startswith("/threshold ") or lower.startswith("threshold "):
        try:
            n = float(message.split()[-1])
            if 0.1 <= n <= 0.9:
                cfg["threshold"] = round(n, 2)
                return _teams_reply(f"📊 Confidence threshold set to **{n}**.")
            return _teams_reply("Threshold must be between 0.1 and 0.9.")
        except ValueError:
            return _teams_reply("Usage: /threshold 0.5")

    return None


HELP_TEXT = """\
⚖️ **Women's Safety Laws — EU RAG Bot**

Grounded in EU legal texts, EIGE reports, and gender-equality research.

**Answer commands** _(always @mention me first)_
🔍 `/ask <question>` — Answer from corpus (+ web fallback)
⚡ `/quick <question>` — Same as /ask

**Settings** _(per conversation)_
⚙️ `/settings` — Show current settings
🌐 `/web on|off` — Toggle live web search
🔢 `/topk N` — Set retrieved passages (3–15)
📊 `/threshold N` — Set confidence threshold (0.1–0.9, default 0.4)

**Knowledge base**
➕ `/add <text>` — Add text to corpus
📄 `/pdf <path>` — Ingest a PDF file

**Utilities**
📊 `/stats` — Corpus statistics
🔄 `/reset` — Clear conversation history
❓ `/help` — This message\
"""

# ---------------------------------------------------------------------------
# Webhook endpoint
# ---------------------------------------------------------------------------

async def webhook(request: web.Request) -> web.Response:
    body = await request.read()
    auth = request.headers.get("Authorization", "")
    if not _verify_hmac(body, auth):
        log.warning("Rejected — HMAC mismatch.")
        return web.json_response({"type": "message",
                                  "text": "⚠️ Auth failed — check TEAMS_SECURITY_TOKEN."})

    try:
        data = await request.json()
    except Exception:
        return _teams_reply("Could not parse request.")

    raw_text: str = (data.get("text") or "").strip()
    message       = _strip_mention(raw_text)
    conv_id: str  = (data.get("conversation") or {}).get("id") or "default"
    user: str     = (data.get("from") or {}).get("name") or "unknown"
    lower         = message.lower().strip()

    log.info("[%s] %s: %s", conv_id[:12], user, message[:80])

    if not message:
        return _teams_reply(HELP_TEXT)

    # ── Settings commands ──────────────────────────────────────────────────
    settings_resp = _handle_settings_cmd(message, lower, conv_id)
    if settings_resp:
        return settings_resp

    # /help
    if lower in ("/help", "help", "?"):
        return _teams_reply(HELP_TEXT)

    # /reset
    if lower in ("/reset", "reset"):
        _history[conv_id].clear()
        return _teams_reply("🔄 Conversation history cleared.")

    # /stats
    if lower in ("/stats", "stats"):
        try:
            data_dir = Path(os.environ.get("PDF_OUTPUT_DIR", "data"))
            chunks_path = data_dir / "chunks.json"
            if chunks_path.exists():
                all_chunks = json.loads(chunks_path.read_text())
                n_chunks   = len(all_chunks)
                n_docs     = len({c["source"] for c in all_chunks})
                return _teams_reply(
                    f"📊 **Corpus Statistics**\n\n"
                    f"- **{n_chunks:,}** chunks indexed\n"
                    f"- **{n_docs}** source documents\n"
                )
        except Exception as exc:
            return _teams_reply(f"Could not load stats: {exc}")

    # /add <text>
    if lower.startswith("/add "):
        content = message[5:].strip()
        if not content:
            return _teams_reply("Please provide text after /add.")
        try:
            msg = add_to_database(text=content, source=f"teams-{date.today().isoformat()}")
            return _teams_reply(f"✅ Added. {msg}")
        except Exception as exc:
            return _teams_reply(f"Error: {exc}")

    # /pdf <path>
    if lower.startswith("/pdf "):
        pdf_path = message[5:].strip()
        if not pdf_path:
            return _teams_reply("Please provide a file path after /pdf.")
        try:
            msg = ingest_pdf(pdf_path=pdf_path)
            return _teams_reply(msg)
        except Exception as exc:
            return _teams_reply(f"Error: {exc}")

    # ── Answer commands (/ask, /quick, or plain question) ──────────────────
    query = None
    if lower.startswith("/ask ") or lower.startswith("ask "):
        query = message.split(" ", 1)[1].strip()
    elif lower.startswith("/quick ") or lower.startswith("quick "):
        query = message.split(" ", 1)[1].strip()
    elif len(message) >= 5:
        query = message

    if not query:
        return _teams_reply(HELP_TEXT)

    _add_history(conv_id, "user", query)
    try:
        answer, sources = await _answer(query, conv_id)
        _add_history(conv_id, "assistant", answer)

        text = answer
        if sources:
            text += "\n\n**Sources:** " + "  ·  ".join(sources)
        return _teams_reply(text)

    except Exception as exc:
        log.error("Answer error: %s", exc, exc_info=True)
        return _teams_reply(f"Error: {exc}")


async def health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def index(request: web.Request) -> web.Response:
    return web.Response(text="Teams RAG bot is running. POST to /webhook.",
                        content_type="text/plain")


# ---------------------------------------------------------------------------
# App + entry point
# ---------------------------------------------------------------------------

app = web.Application()
app.router.add_post("/webhook", webhook)
app.router.add_get("/health",   health)
app.router.add_get("/",         index)


def _prewarm() -> None:
    try:
        log.info("Pre-warming artifacts...")
        _get_artifacts()
        log.info("Pre-warm complete — %d chunks ready.", len(_chunks))
    except Exception as exc:
        log.warning("Pre-warm failed (non-fatal): %s", exc)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    log.info("Teams outgoing webhook server — port %d", PORT)
    if not SECURITY_TOKEN:
        log.warning("TEAMS_SECURITY_TOKEN not set — HMAC auth disabled.")
    _prewarm()
    web.run_app(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
