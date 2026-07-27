from __future__ import annotations

"""
app/observability.py

LLM observability for the Women's Safety RAG system.

There are three things you must be able to see in an LLM system:

  1. Tracing the "thought process" — the exact sequence of sub-agents, tool
     calls, and prompts triggered by a single user request.
  2. Token & cost tracking — the financial cost and latency of every
     interaction, in real time.
  3. Context debugging — exactly what data was injected into the prompt at
     the moment the model hallucinated.

This module provides a dependency-free tracer that covers all three. Traces
are nested spans written to `traces/<trace_id>.json`, with a human-readable
waterfall printable to the terminal.

It composes with `scripts/cost_function.track_cost`, which owns the durable
budget ledger in `cost_tracker.json`; this module adds the per-request view.

Usage:
    from app.observability import tracer

    with tracer.trace("user request", query=q) as t:
        with tracer.span("agent", name="internal_researcher"):
            ...
        print(t.summary())
"""

import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger(__name__)

TRACE_DIR = Path(os.environ.get("TRACE_DIR", "traces"))

#: Prompt/response bodies are truncated in the persisted trace so a long run
#: does not produce a 50 MB JSON file. Raise for deep context debugging.
MAX_CAPTURE_CHARS = int(os.environ.get("TRACE_MAX_CHARS", "4000"))


# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------

@dataclass
class Span:
    """One unit of work: an agent turn, an LLM call, a tool call, retrieval."""

    span_id: str
    trace_id: str
    parent_id: str | None
    kind: str                     # "agent" | "llm" | "tool" | "retrieval" | "root"
    name: str
    started_at: float
    ended_at: float | None = None
    status: str = "running"       # "running" | "ok" | "error"
    error: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    # token / cost accounting for this span alone
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def duration_ms(self) -> float:
        end = self.ended_at if self.ended_at is not None else time.perf_counter()
        return (end - self.started_at) * 1000.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "kind": self.kind,
            "name": self.name,
            "status": self.status,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "attributes": self.attributes,
            "events": self.events,
        }


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------

@dataclass
class Trace:
    """All spans produced by one user request."""

    trace_id: str
    label: str
    started_at: float
    metadata: dict[str, Any] = field(default_factory=dict)
    spans: list[Span] = field(default_factory=list)
    ended_at: float | None = None

    @property
    def duration_ms(self) -> float:
        end = self.ended_at if self.ended_at is not None else time.perf_counter()
        return (end - self.started_at) * 1000.0

    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.spans)

    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.spans)

    def by_kind(self, kind: str) -> list[Span]:
        return [s for s in self.spans if s.kind == kind]

    def top_level_tools(self) -> list[Span]:
        """Tool spans that are not nested inside another tool span.

        A tool invoked through the agent dispatcher opens a span, and the MCP
        tool it calls opens its own child span. Both are useful in the
        waterfall, but counting them naively double-counts every call — so
        summaries count only the outermost one.
        """
        by_id = {s.span_id: s for s in self.spans}
        outermost: list[Span] = []
        for s in self.spans:
            if s.kind != "tool":
                continue
            parent = by_id.get(s.parent_id) if s.parent_id else None
            while parent is not None and parent.kind != "tool":
                parent = by_id.get(parent.parent_id) if parent.parent_id else None
            if parent is None:
                outermost.append(s)
        return outermost

    # -- rendering ---------------------------------------------------------

    def waterfall(self) -> str:
        """Indented call tree — the 'thought process' view."""
        children: dict[str | None, list[Span]] = {}
        for s in self.spans:
            children.setdefault(s.parent_id, []).append(s)

        lines: list[str] = []
        glyphs = {
            "root": "◆", "agent": "▸", "llm": "✦",
            "tool": "⚙", "retrieval": "⌕",
        }

        def walk(parent: str | None, depth: int) -> None:
            for s in children.get(parent, []):
                mark = "✗" if s.status == "error" else " "
                cost = f"${s.cost_usd:.6f}" if s.cost_usd else "—"
                tok = f"{s.total_tokens}t" if s.total_tokens else "—"
                lines.append(
                    f"{mark} {'  ' * depth}{glyphs.get(s.kind, '·')} "
                    f"{s.name:<34.34} {s.duration_ms:>9.1f}ms  {tok:>8}  {cost:>11}"
                )
                if s.error:
                    lines.append(f"{'  ' * (depth + 1)}   ↳ error: {s.error[:100]}")
                walk(s.span_id, depth + 1)

        walk(None, 0)
        return "\n".join(lines)

    def summary(self) -> str:
        agents = self.by_kind("agent")
        llm = self.by_kind("llm")
        tools = self.top_level_tools()
        errors = [s for s in self.spans if s.status == "error"]

        tool_counts: dict[str, int] = {}
        for s in tools:
            label = s.name.removeprefix("dispatch:")
            tool_counts[label] = tool_counts.get(label, 0) + 1
        tool_line = ", ".join(f"{k}×{v}" for k, v in sorted(tool_counts.items())) or "none"

        head = (
            f"\n{'=' * 78}\n"
            f"TRACE {self.trace_id}  —  {self.label}\n"
            f"{'=' * 78}\n"
            f"{self.waterfall()}\n"
            f"{'-' * 78}\n"
            f"  wall time     {self.duration_ms / 1000:.2f}s\n"
            f"  agents        {len(agents)}\n"
            f"  llm calls     {len(llm)}\n"
            f"  tool calls    {len(tools)}  ({tool_line})\n"
            f"  tokens        {self.total_tokens:,}\n"
            f"  cost          ${self.total_cost_usd:.6f}\n"
            f"  errors        {len(errors)}\n"
            f"{'=' * 78}"
        )
        return head

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "label": self.label,
            "started_at": datetime.fromtimestamp(
                time.time() - self.duration_ms / 1000, tz=timezone.utc
            ).isoformat(),
            "duration_ms": round(self.duration_ms, 2),
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "metadata": self.metadata,
            "spans": [s.to_dict() for s in self.spans],
        }

    def save(self, directory: Path | None = None) -> Path:
        directory = directory or TRACE_DIR
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.trace_id}.json"
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------

class Tracer:
    """Thread-local span stack over a single active trace."""

    def __init__(self, *, enabled: bool = True, auto_save: bool = True) -> None:
        self.enabled = enabled
        self.auto_save = auto_save
        self._local = threading.local()
        self._traces: list[Trace] = []
        self._lock = threading.Lock()

    # -- state -------------------------------------------------------------

    @property
    def current_trace(self) -> Trace | None:
        return getattr(self._local, "trace", None)

    @property
    def _stack(self) -> list[Span]:
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    @property
    def current_span(self) -> Span | None:
        return self._stack[-1] if self._stack else None

    @property
    def traces(self) -> list[Trace]:
        with self._lock:
            return list(self._traces)

    def reset(self) -> None:
        with self._lock:
            self._traces.clear()
        self._local.trace = None
        self._local.stack = []

    # -- trace / span context managers ------------------------------------

    @contextmanager
    def trace(self, label: str, **metadata: Any) -> Iterator[Trace]:
        """Open a new root trace for one user request."""
        tr = Trace(
            trace_id=f"tr_{uuid.uuid4().hex[:12]}",
            label=label,
            started_at=time.perf_counter(),
            metadata=metadata,
        )
        prev_trace, prev_stack = self.current_trace, list(self._stack)
        self._local.trace = tr
        self._local.stack = []
        with self._lock:
            self._traces.append(tr)
        try:
            yield tr
        finally:
            tr.ended_at = time.perf_counter()
            if self.auto_save and self.enabled:
                try:
                    tr.save()
                except OSError as exc:
                    log.debug("Could not persist trace: %s", exc)
            self._local.trace = prev_trace
            self._local.stack = prev_stack

    @contextmanager
    def span(self, kind: str, name: str, **attributes: Any) -> Iterator[Span]:
        """Open a child span under the current span (or trace root)."""
        tr = self.current_trace
        if tr is None or not self.enabled:
            # No active trace — yield a detached span so callers never break.
            yield Span(
                span_id="detached", trace_id="none", parent_id=None,
                kind=kind, name=name, started_at=time.perf_counter(),
            )
            return

        sp = Span(
            span_id=f"sp_{uuid.uuid4().hex[:10]}",
            trace_id=tr.trace_id,
            parent_id=self.current_span.span_id if self.current_span else None,
            kind=kind,
            name=name,
            started_at=time.perf_counter(),
            attributes=dict(attributes),
        )
        tr.spans.append(sp)
        self._stack.append(sp)
        try:
            yield sp
            sp.status = "ok"
        except Exception as exc:
            sp.status = "error"
            sp.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            sp.ended_at = time.perf_counter()
            self._stack.pop()

    # -- recording helpers -------------------------------------------------

    def record_llm_call(
        self,
        span: Span,
        *,
        model: str,
        messages: list | None = None,
        response_text: str | None = None,
        usage: Any = None,
        cost_usd: float = 0.0,
    ) -> None:
        """Attach prompt, response, tokens, and cost to an LLM span.

        `messages` is captured (truncated) so that when the model hallucinates
        you can see the exact context that was in the window at that moment.
        """
        span.attributes["model"] = model
        if usage is not None:
            span.prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            span.completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        span.cost_usd = float(cost_usd or 0.0)

        if messages is not None:
            span.attributes["context"] = _capture_messages(messages)
            span.attributes["context_chars"] = sum(
                len(str(_content_of(m))) for m in messages
            )
        if response_text is not None:
            span.attributes["response"] = _truncate(response_text)

    def record_retrieval(
        self,
        span: Span,
        *,
        query: str,
        results: list[dict],
    ) -> None:
        """Capture what retrieval actually returned — context debugging."""
        span.attributes["query"] = _truncate(query, 500)
        span.attributes["n_results"] = len(results)
        span.attributes["retrieved"] = [
            {
                "source": r.get("source"),
                "page": r.get("page"),
                "score": round(float(r.get("score", 0.0)), 4),
                "preview": _truncate(str(r.get("text", "")), 240),
            }
            for r in results[:12]
        ]
        if results:
            scores = [float(r.get("score", 0.0)) for r in results]
            span.attributes["top_score"] = round(max(scores), 4)
            span.attributes["mean_score"] = round(sum(scores) / len(scores), 4)

    def event(self, message: str, **data: Any) -> None:
        """Attach a point-in-time note to the current span."""
        sp = self.current_span
        if sp is None:
            return
        sp.events.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "message": message,
            **{k: _truncate(str(v), 500) for k, v in data.items()},
        })

    def annotate(self, **attributes: Any) -> None:
        sp = self.current_span
        if sp is not None:
            sp.attributes.update(attributes)


# ---------------------------------------------------------------------------
# Cost helper — mirrors gpt-4o-mini / text-embedding-3-small pricing
# ---------------------------------------------------------------------------

_PRICING_PER_TOKEN: dict[str, tuple[float, float]] = {
    # model: (input $/token, output $/token)
    "gpt-4o-mini": (0.15 / 1_000_000, 0.60 / 1_000_000),
    "gpt-4o": (2.50 / 1_000_000, 10.00 / 1_000_000),
    "text-embedding-3-small": (0.02 / 1_000_000, 0.0),
    "text-embedding-3-large": (0.13 / 1_000_000, 0.0),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int = 0) -> float:
    """USD cost for one call.

    Locally served models cost nothing, so they are priced at zero rather than
    being charged hosted rates. Unknown hosted models fall back to gpt-4o-mini.
    """
    from app import llm as _llm

    if _llm.is_ollama():
        return 0.0
    rate_in, rate_out = _PRICING_PER_TOKEN.get(
        model, _PRICING_PER_TOKEN["gpt-4o-mini"]
    )
    return prompt_tokens * rate_in + completion_tokens * rate_out


# ---------------------------------------------------------------------------
# Capture helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, limit: int | None = None) -> str:
    limit = limit if limit is not None else MAX_CAPTURE_CHARS
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f"…[+{len(text) - limit} chars]"


def _content_of(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content", "")
    return getattr(message, "content", "") or ""


def _capture_messages(messages: list) -> list[dict]:
    """Snapshot the prompt window, redacting secrets before it hits disk."""
    from app.security import guard  # local import avoids a cycle at import time

    captured: list[dict] = []
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "?")
        content = _content_of(m)
        if not isinstance(content, str):
            content = json.dumps(content, default=str, ensure_ascii=False)
        safe = guard.redact(content, origin="trace_capture").sanitised
        entry: dict[str, Any] = {"role": role, "content": _truncate(safe)}
        tool_calls = (
            m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", None)
        )
        if tool_calls:
            entry["tool_calls"] = [
                getattr(tc.function, "name", "?") if hasattr(tc, "function")
                else str(tc)
                for tc in tool_calls
            ]
        captured.append(entry)
    return captured


#: Process-wide tracer used by the agents, the MCP server, and the bots.
tracer = Tracer(
    enabled=os.environ.get("TRACING_ENABLED", "1").lower() not in ("0", "false", "no"),
)


__all__ = ["tracer", "Tracer", "Trace", "Span", "estimate_cost", "TRACE_DIR"]
