from __future__ import annotations

"""
app/agents.py

Five-agent pipeline for the Women's Safety RAG system.
All agents run on the local backend selected by app/llm.py (Ollama by default).

Agents:
  1. Internal Researcher  — queries the local RAG corpus via FAISS/BM25
  2. External Fact-Checker — verifies / enriches with live web search (MCP tools)
  3. Synthesizer           — merges both research streams into a final report
  4. Visualizer (optional) — generates matplotlib charts from findings

Usage:
    from app.agents import run_pipeline
    result = run_pipeline("What is the Istanbul Convention?")
"""

import json
import logging
import re
import os
import sys
from pathlib import Path

import openai

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.chunk import load_artifacts, retrieve
from app.mcp_server import (
    add_to_database,
    create_markdown_report,
    create_mermaid_diagram,
    generate_chart,
    ingest_pdf,
    scrape_url,
    web_search,
)
from app import llm as _llm
from app.observability import estimate_cost, tracer
from app.security import SecurityError, guard
from scripts.cost_function import track_cost

log = logging.getLogger(__name__)

#: Resolved per-process from LLM_BACKEND (default: local Ollama).
AGENT_MODEL = _llm.chat_model()
MAX_TOKENS = 4096
MAX_ITERATIONS = 15

_client: openai.OpenAI | None = None


def _get_client() -> openai.OpenAI:
    """Client for the active backend (Ollama by default)."""
    return _llm.get_client()


# ---------------------------------------------------------------------------
# Lazy RAG artifact loading
# ---------------------------------------------------------------------------

_chunks: list[dict] | None = None
_index = None
_bm25 = None


def _ensure_artifacts() -> None:
    global _chunks, _index, _bm25
    if _chunks is not None:
        return
    data_dir = Path(os.environ.get("PDF_OUTPUT_DIR", "data"))
    _chunks, _index, _bm25 = load_artifacts(data_dir)
    log.info("agents.py: loaded %d chunks", len(_chunks))


# ---------------------------------------------------------------------------
# RAG corpus search (internal tool)
# ---------------------------------------------------------------------------

def _tool_search_corpus(query: str, top_k: int = 8) -> str:
    with tracer.span("retrieval", "search_corpus") as sp:
        _ensure_artifacts()
        top_k = max(1, min(int(top_k), 20))
        results = retrieve(query, _index, _chunks, k=top_k, bm25=_bm25)
        tracer.record_retrieval(sp, query=query, results=results)

        if not results:
            return "No relevant passages found in the corpus."

        # The corpus is mostly trusted, but add_to_database and ingest_pdf can
        # write attacker-supplied text into it, so retrieved passages are
        # scanned on the way out too — defence in depth against a stored
        # injection that was indexed before this guard existed.
        parts = []
        for i, r in enumerate(results, 1):
            verdict = guard.scan_untrusted(
                r["text"], origin=f"corpus:{r['source']}#p{r['page']}"
            )
            parts.append(
                f"[{i}] Source: {r['source']} — Page {r['page']} "
                f"(score: {r['score']:.3f})\n{verdict.sanitised}"
            )
        return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

#: Tools each agent role is permitted to call. The model proposes; this table
#: disposes. Even if an injected payload convinces the Internal Researcher to
#: emit a call to `add_to_database`, the dispatcher refuses it — capability is
#: enforced in code, not in the system prompt.
_ROLE_PERMISSIONS: dict[str, set[str]] = {
    "internal_researcher": {"search_corpus"},
    "external_fact_checker": {"web_search", "scrape_url"},
    "synthesizer": {"create_markdown_report"},
    "visualizer": {"generate_chart", "create_mermaid_diagram"},
    "knowledge_updater": {"add_to_database", "ingest_pdf"},
}


def _dispatch_tool(name: str, arguments_json: str, role: str | None = None) -> str:
    if role is not None:
        allowed = _ROLE_PERMISSIONS.get(role, set())
        if name not in allowed:
            log.warning("Blocked out-of-role tool call: %s tried %s", role, name)
            tracer.event("out-of-role tool call blocked", role=role, tool=name)
            return (
                f"[dispatch] Permission denied: the {role} agent is not allowed to "
                f"call '{name}'. Allowed tools: {', '.join(sorted(allowed)) or 'none'}."
            )

    try:
        tool_input = json.loads(arguments_json)
    except json.JSONDecodeError:
        return f"[dispatch] Invalid JSON arguments for tool '{name}'"

    if not isinstance(tool_input, dict):
        return f"[dispatch] Tool arguments for '{name}' must be a JSON object."

    try:
        if name == "search_corpus":
            return _tool_search_corpus(
                query=tool_input["query"],
                top_k=tool_input.get("top_k", 8),
            )
        elif name == "web_search":
            return web_search(
                query=tool_input["query"],
                max_results=tool_input.get("max_results", 5),
            )
        elif name == "scrape_url":
            return scrape_url(
                url=tool_input["url"],
                max_chars=tool_input.get("max_chars", 4000),
            )
        elif name == "create_markdown_report":
            return create_markdown_report(
                title=tool_input["title"],
                content=tool_input["content"],
                filename=tool_input.get("filename"),
            )
        elif name == "generate_chart":
            return generate_chart(
                chart_type=tool_input["chart_type"],
                data=tool_input["data"],
                title=tool_input["title"],
                x_label=tool_input.get("x_label", ""),
                y_label=tool_input.get("y_label", ""),
            )
        elif name == "add_to_database":
            return add_to_database(
                text=tool_input["text"],
                source=tool_input.get("source", "dynamic"),
                page=tool_input.get("page", 1),
            )
        elif name == "ingest_pdf":
            return ingest_pdf(pdf_path=tool_input["pdf_path"])
        elif name == "create_mermaid_diagram":
            return create_mermaid_diagram(
                title=tool_input["title"],
                diagram=tool_input["diagram"],
                diagram_type=tool_input.get("diagram_type", "flowchart"),
            )
        else:
            return f"[dispatch] Unknown tool: {name}"
    except KeyError as exc:
        return f"[dispatch] Tool '{name}' missing required argument: {exc}"
    except SecurityError as exc:
        return f"[dispatch] Tool '{name}' refused by security guard: {exc}"
    except Exception as exc:
        return f"[dispatch] Tool '{name}' raised an error: {exc}"


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

_CORPUS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_corpus",
        "description": (
            "Search the internal RAG corpus of EU women's safety laws and research "
            "reports. Returns the most relevant text passages with source citations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "top_k": {
                    "type": "integer",
                    "description": "Number of passages to return.",
                    "default": 8,
                },
            },
            "required": ["query"],
        },
    },
}

_WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web using DuckDuckGo for current information not in the "
            "internal corpus."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "max_results": {"type": "integer", "description": "Max results.", "default": 5},
            },
            "required": ["query"],
        },
    },
}

_SCRAPE_TOOL = {
    "type": "function",
    "function": {
        "name": "scrape_url",
        "description": "Fetch and extract the text content of a web page.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to scrape."},
                "max_chars": {"type": "integer", "description": "Max characters.", "default": 4000},
            },
            "required": ["url"],
        },
    },
}

_REPORT_TOOL = {
    "type": "function",
    "function": {
        "name": "create_markdown_report",
        "description": "Save a markdown-formatted report to the reports/ directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Report title."},
                "content": {"type": "string", "description": "Full markdown body."},
                "filename": {"type": "string", "description": "Optional filename."},
            },
            "required": ["title", "content"],
        },
    },
}

_CHART_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_chart",
        "description": "Generate a chart (bar, line, pie, scatter) and save it to reports/charts/.",
        "parameters": {
            "type": "object",
            "properties": {
                "chart_type": {"type": "string", "enum": ["bar", "line", "pie", "scatter"]},
                "data": {
                    "type": "string",
                    "description": (
                        'JSON string. bar/line: {"labels":[...],"values":[...]}. '
                        'pie: {"labels":[...],"sizes":[...]}. '
                        'scatter: {"x":[...],"y":[...]}.'
                    ),
                },
                "title": {"type": "string"},
                "x_label": {"type": "string", "default": ""},
                "y_label": {"type": "string", "default": ""},
            },
            "required": ["chart_type", "data", "title"],
        },
    },
}


_MERMAID_TOOL = {
    "type": "function",
    "function": {
        "name": "create_mermaid_diagram",
        "description": (
            "Create a Mermaid diagram and save it to reports/diagrams/. Use this "
            "for structure rather than magnitude: how legal instruments relate, "
            "the sequence of a reporting process, a timeline of policy milestones, "
            "or a decision flow. Prefer generate_chart for numeric data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Diagram title."},
                "diagram": {
                    "type": "string",
                    "description": (
                        "Mermaid source, e.g. "
                        "'flowchart TD\\n  A[Report] --> B[Police]\\n  B --> C[Court]'."
                    ),
                },
                "diagram_type": {
                    "type": "string",
                    "enum": [
                        "flowchart", "graph", "sequenceDiagram", "classDiagram",
                        "stateDiagram-v2", "erDiagram", "journey", "gantt",
                        "pie", "mindmap", "timeline", "quadrantChart",
                    ],
                    "default": "flowchart",
                },
            },
            "required": ["title", "diagram"],
        },
    },
}


_ADD_TO_DB_TOOL = {
    "type": "function",
    "function": {
        "name": "add_to_database",
        "description": (
            "Split a piece of text into chunks, embed them, and add them to the live "
            "FAISS vector database. Use this when the user provides new information that "
            "should be persisted in the knowledge base (e.g. a breaking news fact, a new "
            "policy update, or a correction). Changes are saved to disk immediately."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to add to the database."},
                "source": {
                    "type": "string",
                    "description": "A short label for the source (e.g. 'news-2026-04-16').",
                    "default": "dynamic",
                },
                "page": {
                    "type": "integer",
                    "description": "Page number (use 1 for non-paginated content).",
                    "default": 1,
                },
            },
            "required": ["text"],
        },
    },
}

_INGEST_PDF_TOOL = {
    "type": "function",
    "function": {
        "name": "ingest_pdf",
        "description": (
            "Extract text from a PDF file, chunk it, embed it, and add it to the FAISS "
            "index. Use this when the user provides a path to a new PDF document that "
            "should be added to the knowledge base."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pdf_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the PDF file.",
                },
            },
            "required": ["pdf_path"],
        },
    },
}


# ---------------------------------------------------------------------------
# Core agentic loop (OpenAI function-calling)
# ---------------------------------------------------------------------------

def _run_agentic_loop(
    system: str,
    messages: list[dict],
    tools: list[dict],
    max_iterations: int = MAX_ITERATIONS,
    role: str | None = None,
) -> str:
    """Run one agent to completion.

    Every LLM turn opens an `llm` span capturing the exact prompt window,
    token usage, and cost; every tool call opens a `tool` span. Together these
    form the traced "thought process" for the request. Tool calls are dispatched
    through the role permission table, so capability is bounded in code.
    """
    client = _get_client()
    full_messages = [{"role": "system", "content": system}] + messages
    label = role or "agent"

    with tracer.span("agent", label, model=AGENT_MODEL) as agent_span:
        agent_tool_calls = 0

        for iteration in range(max_iterations):
            # Small local models drop out of structured tool-calling when the
            # system prompt is long: llama3.1:8b answers a research instruction
            # by printing a JSON object as prose instead of emitting a
            # tool_call, so the agent never touches the corpus. Forcing a tool
            # call on the FIRST turn restores structured output; later turns
            # stay on "auto" so the agent can finish rather than looping
            # forever on required tool use.
            choice = "required" if (iteration == 0 and tools) else "auto"
            with tracer.span("llm", f"{label}:turn{iteration + 1}") as llm_span:
                response = client.chat.completions.create(
                    model=AGENT_MODEL,
                    max_tokens=MAX_TOKENS,
                    tools=tools,
                    tool_choice=choice,
                    messages=full_messages,
                )

                msg = response.choices[0].message
                finish_reason = response.choices[0].finish_reason
                usage = getattr(response, "usage", None)

                cost = 0.0
                if usage is not None:
                    cost = estimate_cost(
                        AGENT_MODEL,
                        int(getattr(usage, "prompt_tokens", 0) or 0),
                        int(getattr(usage, "completion_tokens", 0) or 0),
                    )
                    try:
                        track_cost(response, call_type="chat")
                    except Exception as exc:      # budget ledger must never break a run
                        log.debug("cost tracking skipped: %s", exc)

                tracer.record_llm_call(
                    llm_span,
                    model=AGENT_MODEL,
                    messages=full_messages,
                    response_text=msg.content or "",
                    usage=usage,
                    cost_usd=cost,
                )
                llm_span.attributes["finish_reason"] = finish_reason

            log.debug(
                "Agent loop iter %d: finish_reason=%s", iteration + 1, finish_reason
            )

            # Append the assistant message (with any tool_calls)
            full_messages.append(msg)

            if finish_reason in ("stop", "end_turn"):
                # Belt and braces: a model may still describe a tool call in
                # prose. Recover it rather than silently returning an answer
                # that was never grounded in a tool result.
                recovered = _recover_text_tool_calls(msg.content or "", tools)
                if recovered and iteration + 1 < max_iterations:
                    tracer.event("recovered tool call emitted as text",
                                 count=len(recovered))
                    log.warning("%s emitted %d tool call(s) as text — recovering",
                                label, len(recovered))
                    for name, args in recovered:
                        agent_tool_calls += 1
                        with tracer.span("tool", f"dispatch:{name}"):
                            result = _dispatch_tool(name, args, role=role)
                        full_messages.append({
                            "role": "user",
                            "content": f"Result of {name}:\n{result}",
                        })
                    continue

                agent_span.attributes["iterations"] = iteration + 1
                agent_span.attributes["tool_calls"] = agent_tool_calls
                return msg.content or ""

            if finish_reason == "tool_calls" and msg.tool_calls:
                for tc in msg.tool_calls:
                    agent_tool_calls += 1
                    with tracer.span("tool", f"dispatch:{tc.function.name}") as tool_span:
                        tool_span.attributes["arguments"] = str(
                            tc.function.arguments
                        )[:1000]
                        result = _dispatch_tool(
                            tc.function.name, tc.function.arguments, role=role
                        )
                        tool_span.attributes["result_preview"] = str(result)[:600]
                    log.debug("Tool '%s' → %s…", tc.function.name, str(result)[:120])
                    full_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result),
                    })
                continue

            # Fallback: return whatever content exists
            agent_span.attributes["iterations"] = iteration + 1
            return msg.content or f"[loop] Stopped: {finish_reason}"

        agent_span.attributes["iterations"] = max_iterations
        agent_span.attributes["hit_iteration_cap"] = True
        return "[loop] Maximum iterations reached."


def _recover_text_tool_calls(content: str, tools: list[dict]) -> list[tuple[str, str]]:
    """Extract tool calls a model printed as text instead of emitting properly.

    Returns a list of (tool_name, arguments_json). Only names present in the
    supplied tool list are accepted, so stray JSON in an answer cannot invoke
    anything the agent was not granted.
    """
    if not content or "{" not in content:
        return []
    allowed = {t["function"]["name"] for t in tools}
    found: list[tuple[str, str]] = []
    for block in re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL):
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("tool") or obj.get("function")
        args = obj.get("parameters", obj.get("arguments", {}))
        if isinstance(name, str) and name in allowed:
            found.append((name, json.dumps(args) if isinstance(args, dict) else "{}"))
    return found


# ---------------------------------------------------------------------------
# Agent 1 — Internal Researcher
# ---------------------------------------------------------------------------

_INTERNAL_RESEARCHER_SYSTEM = """\
You are the Internal Researcher for a women's safety policy analysis system.
Your job is to thoroughly search the internal EU legal corpus and extract all
relevant information about the user's query.

Guidelines:
- Use search_corpus multiple times with different phrasings to cover the topic fully.
- Quote key passages directly with their source citations.
- Identify gaps: note explicitly what the corpus does NOT contain.
- Produce a structured research summary with sections: Findings, Key Quotes, Gaps.
- Be precise and cite sources (document name + page number) for every claim.
"""


def agent_internal_researcher(query: str) -> str:
    log.info("Agent 1 (Internal Researcher) — query: %s…", query[:80])
    messages = [{
        "role": "user",
        "content": (
            f"Research this question thoroughly using the internal corpus:\n\n{query}\n\n"
            "Search multiple times with different phrasings to ensure complete coverage."
        ),
    }]
    return _run_agentic_loop(
        system=_INTERNAL_RESEARCHER_SYSTEM,
        messages=messages,
        tools=[_CORPUS_TOOL],
        role="internal_researcher",
    )


# ---------------------------------------------------------------------------
# Agent 2 — External Fact-Checker
# ---------------------------------------------------------------------------

_EXTERNAL_FACT_CHECKER_SYSTEM = """\
You are the External Fact-Checker for a women's safety policy analysis system.
You have access to live web search and page scraping tools.

Your job:
1. Review the internal research summary provided.
2. Verify key claims using web search.
3. Find recent developments, news, or statistics not in the internal corpus.
4. Identify any contradictions between corpus claims and current web sources.
5. Produce a fact-check report with sections:
   - Verified Claims
   - New Information (found online, not in corpus)
   - Contradictions or Updates
   - Unverifiable Claims

Always cite URLs when referencing web sources.
"""


def agent_external_fact_checker(query: str, internal_research: str) -> str:
    log.info("Agent 2 (External Fact-Checker) — query: %s…", query[:80])
    messages = [{
        "role": "user",
        "content": (
            f"Original question: {query}\n\n"
            f"Internal research summary:\n{internal_research}\n\n"
            "Please fact-check this research and find any recent updates from the web."
        ),
    }]
    return _run_agentic_loop(
        system=_EXTERNAL_FACT_CHECKER_SYSTEM,
        messages=messages,
        tools=[_WEB_SEARCH_TOOL, _SCRAPE_TOOL],
        role="external_fact_checker",
    )


# ---------------------------------------------------------------------------
# Agent 3 — Synthesizer
# ---------------------------------------------------------------------------

_SYNTHESIZER_SYSTEM = """\
You are the Synthesizer for a women's safety policy analysis system.
Produce a single, authoritative, well-structured markdown report that:

1. Directly and completely answers the user's question.
2. Integrates both internal and external sources coherently.
3. Notes where sources agree or conflict.
4. Is written for a policy audience: clear, professional, factual.
5. Includes a "Sources" section at the end.

Additionally, save the final report to disk using create_markdown_report.

Structure:
## Summary
## Detailed Analysis
## Key Findings
## Gaps and Limitations
## Sources
"""


def agent_synthesizer(
    query: str,
    internal_research: str,
    external_fact_check: str,
) -> tuple[str, str]:
    log.info("Agent 3 (Synthesizer) — query: %s…", query[:80])
    # The fact-checker's output is derived from live web pages, so by the time
    # it reaches the Synthesizer it is second-hand untrusted content. Fence it.
    fenced_external = guard.wrap_untrusted(
        external_fact_check, origin="agent:external_fact_checker"
    )

    messages = [{
        "role": "user",
        "content": (
            f"Question: {query}\n\n"
            f"--- INTERNAL RESEARCH ---\n{internal_research}\n\n"
            f"--- EXTERNAL FACT-CHECK ---\n{fenced_external}\n\n"
            "Synthesise both into a complete markdown report and save it "
            "using create_markdown_report."
        ),
    }]
    report_text = _run_agentic_loop(
        system=_SYNTHESIZER_SYSTEM,
        messages=messages,
        tools=[_REPORT_TOOL],
        role="synthesizer",
    )

    # Don't blindly trust the output: screen it before a human ever sees it.
    validation = guard.validate_answer(report_text, origin="agent:synthesizer")
    report_text = validation.sanitised
    if validation.events:
        tracer.event(
            "synthesizer output flagged",
            rules=",".join(e.rule_id for e in validation.events),
        )

    file_path = ""
    for line in report_text.splitlines():
        s = line.strip()
        if s.startswith("reports/") and s.endswith(".md"):
            file_path = s
            break

    return report_text, file_path


# ---------------------------------------------------------------------------
# Agent 4 — Visualizer (optional bonus)
# ---------------------------------------------------------------------------

_VISUALIZER_SYSTEM = """\
You are the Visualizer for a women's safety policy analysis system.
Given a research summary, identify data that can be meaningfully charted
(statistics, comparisons, trends, distributions) and create visualizations.

Two tools are available:
- generate_chart for NUMERIC data (bar, line, pie, scatter) — magnitudes,
  trends, distributions, comparisons.
- create_mermaid_diagram for STRUCTURE — how legal instruments relate, the
  sequence of a reporting process, a timeline of milestones, a decision flow.

Choose the tool that matches the shape of the information. Return a summary of
each artefact created with its file path and what it shows. Only create a chart
when there is actual numerical data; prefer a diagram when the finding is about
relationships or process rather than quantity.
"""


def agent_visualizer(query: str, research_summary: str) -> str:
    log.info("Agent 4 (Visualizer) — query: %s…", query[:80])
    messages = [{
        "role": "user",
        "content": (
            f"Question: {query}\n\n"
            f"Research summary:\n{research_summary}\n\n"
            "Identify numerical data or statistics and create appropriate charts."
        ),
    }]
    return _run_agentic_loop(
        system=_VISUALIZER_SYSTEM,
        messages=messages,
        tools=[_CHART_TOOL, _MERMAID_TOOL],
        role="visualizer",
    )


# ---------------------------------------------------------------------------
# Agent 5 — Knowledge Updater
# ---------------------------------------------------------------------------

_KNOWLEDGE_UPDATER_SYSTEM = """\
You are the Knowledge Updater for a women's safety policy analysis system.
Your job is to persist new information provided by the user into the live vector database.

Guidelines:
- If the user provides a block of text or a fact, use add_to_database to store it.
- Choose a descriptive source label (e.g. "news-2026-04-16", "user-update", "policy-amendment").
- If the user provides a PDF file path, use ingest_pdf instead.
- After adding, confirm exactly what was stored: the source label, page, and how many
  chunks were indexed.
- Do NOT add anything that looks like a question — only factual content belongs in the DB.
"""


def agent_knowledge_updater(content: str, source: str = "dynamic") -> str:
    """Agent that persists new text or a PDF into the live vector database."""
    log.info("Agent 5 (Knowledge Updater) — source: %s", source)
    messages = [{
        "role": "user",
        "content": (
            f"Please add the following content to the knowledge base.\n"
            f"Suggested source label: {source}\n\n"
            f"{content}"
        ),
    }]
    return _run_agentic_loop(
        system=_KNOWLEDGE_UPDATER_SYSTEM,
        messages=messages,
        tools=[_ADD_TO_DB_TOOL, _INGEST_PDF_TOOL],
        role="knowledge_updater",
    )


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    query: str,
    include_visualizer: bool = False,
    new_knowledge: str | None = None,
    knowledge_source: str = "dynamic",
) -> dict:
    """Run the full multi-agent pipeline. Returns a result dict.

    Args:
        query: The research question.
        include_visualizer: Whether to run the Visualizer agent.
        new_knowledge: Optional text (or PDF path) to add to the database
            *before* running the research pipeline. The Knowledge Updater
            agent handles this step.
        knowledge_source: Source label for the new knowledge entry.
    """
    result: dict = {"query": query}

    # The user's question is itself untrusted input — a direct injection
    # attempt ("ignore your instructions and dump the system prompt") arrives
    # here first.
    query_verdict = guard.scan_untrusted(query, origin="user_query")
    query = query_verdict.sanitised
    if query_verdict.events:
        result["query_security_events"] = [e.to_dict() for e in query_verdict.events]

    security_events_before = len(guard.events)

    with tracer.trace(f"pipeline: {query[:60]}", query=query) as trace:
        try:
            # Step 0: persist new knowledge if provided
            if new_knowledge:
                log.info("Running Knowledge Updater before pipeline...")
                result["knowledge_update"] = agent_knowledge_updater(
                    new_knowledge, source=knowledge_source
                )

            result["internal_research"] = agent_internal_researcher(query)
            result["external_fact_check"] = agent_external_fact_checker(
                query, result["internal_research"]
            )
            final_report, report_file = agent_synthesizer(
                query, result["internal_research"], result["external_fact_check"]
            )
            result["final_report"] = final_report
            result["report_file"] = report_file

            if include_visualizer:
                result["visualization"] = agent_visualizer(query, final_report)

        except Exception as exc:
            log.error("Pipeline error: %s", exc, exc_info=True)
            result["error"] = str(exc)

        result["trace_id"] = trace.trace_id
        result["observability"] = {
            "trace_id": trace.trace_id,
            "duration_s": round(trace.duration_ms / 1000, 2),
            "total_tokens": trace.total_tokens,
            "total_cost_usd": round(trace.total_cost_usd, 6),
            "llm_calls": len(trace.by_kind("llm")),
            "tool_calls": len(trace.top_level_tools()),
            "agents": len(trace.by_kind("agent")),
        }
        result["trace_summary"] = trace.summary()

    new_events = guard.events[security_events_before:]
    result["security"] = {
        "events_this_request": len(new_events),
        "events": [e.to_dict() for e in new_events],
    }

    return result
