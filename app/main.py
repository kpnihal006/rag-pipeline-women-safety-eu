from __future__ import annotations

"""
app/main.py

HITL (Human-in-the-Loop) supervisor for the Women's Safety RAG pipeline.
Runs on the local backend selected by app/llm.py (Ollama by default).

Flow:
  1. Supervisor LLM routes: "corpus_only" or "full_pipeline"
  2. Pipeline runs
  3. Human reviews draft (approve / edit / re-research / rewrite / publish)
  4. Approved reports saved to reports/approved/ and optionally Slack

Usage:
    uv run python app/main.py --query "What is the Istanbul Convention?"
    uv run python app/main.py --query "..." --auto-approve
    uv run python app/main.py --query "..." --visualizer
    uv run python app/main.py --query "..." --force-route full_pipeline
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import openai
import requests as http_requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agents import agent_knowledge_updater, run_pipeline
from app import llm as _llm
from app.observability import tracer
from app.security import guard

load_dotenv()

log = logging.getLogger(__name__)

#: Resolved per-process from LLM_BACKEND (default: local Ollama).
AGENT_MODEL = _llm.chat_model()
APPROVED_DIR = Path("reports") / "approved"
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")

_supervisor_client: openai.OpenAI | None = None


def _get_client() -> openai.OpenAI:
    """Client for the active backend (Ollama by default)."""
    return _llm.get_client()


# ---------------------------------------------------------------------------
# Supervisor: route query
# ---------------------------------------------------------------------------

_ROUTER_SYSTEM = """\
You are a routing supervisor for a women's safety policy Q&A system.
Classify the incoming query and return a JSON object with exactly two keys:
  "route": either "corpus_only" or "full_pipeline"
  "reason": one sentence explaining the choice

Routing rules:
- "corpus_only"  : factual, well-scoped questions answerable from EU legal texts.
- "full_pipeline": questions needing synthesis, current events, or external verification.

Return ONLY valid JSON, no prose.
"""


def supervisor_route(query: str, force_route: str | None = None) -> str:
    if force_route in ("corpus_only", "full_pipeline"):
        log.info("Route forced to: %s", force_route)
        return force_route

    client = _get_client()
    response = client.chat.completions.create(
        model=AGENT_MODEL,
        max_tokens=128,
        temperature=0,
        messages=[
            {"role": "system", "content": _ROUTER_SYSTEM},
            {"role": "user", "content": query},
        ],
    )
    raw = response.choices[0].message.content or ""
    try:
        parsed = json.loads(raw)
        route = parsed.get("route", "full_pipeline")
        reason = parsed.get("reason", "")
        log.info("Supervisor route: %s — %s", route, reason)
        return route if route in ("corpus_only", "full_pipeline") else "full_pipeline"
    except json.JSONDecodeError:
        log.warning("Could not parse supervisor JSON: %r — defaulting to full_pipeline", raw)
        return "full_pipeline"


# ---------------------------------------------------------------------------
# corpus_only path
# ---------------------------------------------------------------------------

def run_corpus_only(query: str) -> dict:
    data_dir = Path(os.environ.get("PDF_OUTPUT_DIR", "data"))
    from scripts.chunk import load_artifacts, retrieve, generate_answer

    with tracer.trace(f"corpus_only: {query[:60]}", query=query, route="corpus_only") as tr:
        chunks, index, bm25 = load_artifacts(data_dir)

        with tracer.span("retrieval", "hybrid_retrieve") as sp:
            results = retrieve(query, index, chunks, bm25=bm25)
            tracer.record_retrieval(sp, query=query, results=results)

        # generate_answer owns its own OpenAI call and books it straight to the
        # ledger, so read the ledger delta to attribute cost to this span.
        from scripts.cost_function import get_summary

        before = get_summary().get("total_usd", 0.0)
        with tracer.span("llm", "generate_answer") as llm_span:
            answer = generate_answer(query, results)
            spent = get_summary().get("total_usd", 0.0) - before
            llm_span.cost_usd = max(0.0, spent)
            llm_span.attributes["cost_source"] = "cost_tracker.json delta"

        validation = guard.validate_answer(
            answer, contexts=[r["text"] for r in results], origin="corpus_only"
        )
        answer = validation.sanitised

        sources, seen = [], set()
        for r in results:
            key = (r["source"], r["page"])
            if key not in seen:
                seen.add(key)
                sources.append(f"- **{r['source']}** — Page {r['page']}")

        final = answer + (
            "\n\n---\n**Sources**\n" + "\n".join(sources) if sources else ""
        )
        observability = {
            "trace_id": tr.trace_id,
            "duration_s": round(tr.duration_ms / 1000, 2),
            "total_tokens": tr.total_tokens,
            "total_cost_usd": round(tr.total_cost_usd, 6),
        }
        summary = tr.summary()

    return {
        "query": query,
        "route": "corpus_only",
        "final_report": final,
        "report_file": "",
        "trace_id": observability["trace_id"],
        "observability": observability,
        "trace_summary": summary,
        "security": {
            "events_this_request": len(validation.events),
            "events": [e.to_dict() for e in validation.events],
        },
    }


# ---------------------------------------------------------------------------
# HITL review loop
# ---------------------------------------------------------------------------

def _print_review_context(result: dict | None) -> None:
    """Show the reviewer the evidence they need: cost, trace, and security."""
    if not result:
        return

    obs = result.get("observability")
    if obs:
        print("\n" + "-" * 60)
        print("OBSERVABILITY")
        print("-" * 60)
        print(f"  trace id    {obs.get('trace_id')}")
        print(f"  wall time   {obs.get('duration_s')}s")
        print(f"  tokens      {obs.get('total_tokens'):,}")
        print(f"  cost        ${obs.get('total_cost_usd', 0):.6f}")
        if "llm_calls" in obs:
            print(
                f"  calls       {obs.get('agents')} agents, "
                f"{obs.get('llm_calls')} llm, {obs.get('tool_calls')} tools"
            )

    sec = result.get("security") or {}
    n = sec.get("events_this_request", 0)
    print("\n" + "-" * 60)
    print(f"SECURITY — {n} guardrail event(s) this request")
    print("-" * 60)
    if n:
        for e in sec.get("events", [])[:15]:
            print(
                f"  [{e['severity'].upper():<6}] {e['category']}/{e['rule_id']}"
                f"  ← {e['origin']}"
            )
            print(f"           {e['excerpt']}")
        print("\n  Review these before approving: content flagged here was "
              "neutralised, but the draft may still reflect it.")
    else:
        print("  No prompt-injection, leakage, or validation issues detected.")


def hitl_review(
    draft: str,
    query: str,
    auto_approve: bool = False,
    result: dict | None = None,
) -> tuple[str, bool]:
    if auto_approve:
        log.info("Auto-approve — skipping human review.")
        return draft, True

    print("\n" + "=" * 60)
    print("DRAFT ANSWER / REPORT")
    print("=" * 60)
    print(draft)
    print("=" * 60)

    _print_review_context(result)

    print("\nOptions:")
    print("  [A] Approve and publish")
    print("  [E] Edit manually")
    print("  [R] Request more research (re-run full pipeline)")
    print("  [W] Request rewrite")
    print("  [T] Show full execution trace (thought process)")
    print("  [S] Show full security report")
    print("  [P] Approve without publishing")
    print("  [Q] Quit / discard")

    while True:
        choice = input("\nYour choice [A/E/R/W/T/S/P/Q]: ").strip().upper()

        if choice == "T":
            print(result.get("trace_summary", "No trace captured.")
                  if result else "No trace captured.")
            continue
        elif choice == "S":
            print(json.dumps(guard.report(), indent=2, ensure_ascii=False))
            continue

        if choice == "A":
            return draft, True
        elif choice == "P":
            return draft, False
        elif choice == "E":
            edited = _open_in_editor(draft)
            sub = input("Approve edited version? [A=publish/P=keep/Q=discard]: ").strip().upper()
            if sub == "A":
                return edited, True
            elif sub == "P":
                return edited, False
            else:
                return draft, False
        elif choice == "R":
            print("Re-running full pipeline...")
            new_result = run_pipeline(query, include_visualizer=False)
            new_draft = new_result.get("final_report", draft)
            return hitl_review(new_draft, query, auto_approve=False, result=new_result)
        elif choice == "W":
            feedback = input("Describe what needs to change: ").strip()
            rewritten = _request_rewrite(draft, query, feedback)
            return hitl_review(rewritten, query, auto_approve=False, result=result)
        elif choice == "Q":
            print("Discarding.")
            return draft, False
        else:
            print("Invalid choice.")


def _open_in_editor(text: str) -> str:
    import subprocess
    import tempfile
    editor = os.environ.get("EDITOR", "nano")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8", prefix="rag_edit_"
    ) as tmp:
        tmp.write(text)
        tmp_path = tmp.name
    try:
        subprocess.call([editor, tmp_path])
        with open(tmp_path, encoding="utf-8") as fh:
            return fh.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _request_rewrite(draft: str, query: str, feedback: str) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=AGENT_MODEL,
        max_tokens=4096,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a policy writing assistant. Rewrite the provided draft "
                    "based on the reviewer's feedback. Preserve accurate facts and "
                    "citations. Return only the rewritten report, no preamble."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original question: {query}\n\n"
                    f"Current draft:\n{draft}\n\n"
                    f"Reviewer feedback: {feedback}\n\n"
                    "Rewrite the report addressing this feedback."
                ),
            },
        ],
    )
    return response.choices[0].message.content or draft


# ---------------------------------------------------------------------------
# Publish
# ---------------------------------------------------------------------------

def _publish_to_file(content: str, query: str) -> str:
    APPROVED_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in query[:40])
    out_path = APPROVED_DIR / f"{ts}_{safe}.md"
    out_path.write_text(f"# {query}\n\n{content}\n", encoding="utf-8")
    log.info("Published → %s", out_path)
    return str(out_path)


def _publish_to_slack(content: str, query: str) -> bool:
    if not SLACK_WEBHOOK_URL:
        log.warning("SLACK_WEBHOOK_URL not set — skipping Slack.")
        return False
    try:
        snippet = content[:3500] + ("…" if len(content) > 3500 else "")
        resp = http_requests.post(
            SLACK_WEBHOOK_URL,
            json={"text": f"*RAG Report: {query}*\n\n{snippet}", "mrkdwn": True},
            timeout=10,
        )
        if resp.ok:
            log.info("Published to Slack.")
            return True
        log.warning("Slack publish failed: %s", resp.status_code)
        return False
    except Exception as exc:
        log.warning("Slack error: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="Women's Safety RAG — HITL Supervisor")
    parser.add_argument("--query", "-q", default=None, help="Question to research.")
    parser.add_argument("--auto-approve", action="store_true", help="Skip human review.")
    parser.add_argument("--visualizer", action="store_true", help="Run Visualizer agent.")
    parser.add_argument(
        "--force-route",
        choices=["corpus_only", "full_pipeline"],
        default=None,
    )
    parser.add_argument("--no-slack", action="store_true")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print the full execution trace (agents, tools, tokens, cost).",
    )
    parser.add_argument(
        "--add",
        metavar="TEXT",
        default=None,
        help="Add TEXT to the knowledge base before running the query.",
    )
    parser.add_argument(
        "--pdf",
        metavar="PATH",
        default=None,
        help="Ingest a PDF file into the knowledge base (no query required).",
    )
    args = parser.parse_args()

    # --pdf: standalone ingest, no query needed
    if args.pdf:
        log.info("Ingesting PDF: %s", args.pdf)
        result_text = agent_knowledge_updater(args.pdf, source=Path(args.pdf).stem)
        print(result_text)
        if not args.query:
            return

    query = args.query.strip() if args.query else ""
    if not query:
        log.error("--query is required unless using --pdf alone.")
        sys.exit(1)

    log.info("Query: %s", query)

    route = supervisor_route(query, force_route=args.force_route)
    log.info("Route: %s", route)

    result = run_corpus_only(query) if route == "corpus_only" else run_pipeline(
        query,
        include_visualizer=args.visualizer,
        new_knowledge=args.add,
        knowledge_source=f"cli-{datetime.now().strftime('%Y%m%d')}",
    )

    if "error" in result:
        log.error("Pipeline error: %s", result["error"])
        sys.exit(1)

    draft = result.get("final_report", "")
    if not draft:
        log.error("Empty report.")
        sys.exit(1)

    final_text, approved = hitl_review(
        draft, query, auto_approve=args.auto_approve, result=result
    )

    if args.trace:
        print(result.get("trace_summary", ""))

    if not approved:
        log.info("Not approved — nothing published.")
        return

    # Last gate before anything leaves the process.
    final_text = guard.redact(final_text, origin="publish").sanitised

    file_path = _publish_to_file(final_text, query)
    print(f"\nPublished to: {file_path}")

    obs = result.get("observability") or {}
    if obs:
        log.info(
            "Run cost $%.6f over %s tokens (trace %s)",
            obs.get("total_cost_usd", 0.0),
            f"{obs.get('total_tokens', 0):,}",
            obs.get("trace_id"),
        )

    if not args.no_slack:
        _publish_to_slack(final_text, query)

    if args.visualizer and result.get("visualization"):
        print("\nVisualization output:")
        print(result["visualization"])


if __name__ == "__main__":
    main()
