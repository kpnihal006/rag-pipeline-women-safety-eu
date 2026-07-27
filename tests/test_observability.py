from __future__ import annotations

"""Tests for app/observability.py — the three things you must be able to
see in an LLM system: the thought process, the cost, and the context."""

import json

import pytest

from app.observability import Tracer, estimate_cost


@pytest.fixture
def tr():
    """An isolated tracer that does not write to disk."""
    return Tracer(enabled=True, auto_save=False)


# ---------------------------------------------------------------------------
# 1. Tracing the thought process
# ---------------------------------------------------------------------------

class TestTracing:

    def test_trace_captures_spans(self, tr):
        with tr.trace("request") as trace:
            with tr.span("agent", "researcher"):
                with tr.span("tool", "search_corpus"):
                    pass
        assert len(trace.spans) == 2
        assert {s.name for s in trace.spans} == {"researcher", "search_corpus"}

    def test_spans_nest_correctly(self, tr):
        with tr.trace("request") as trace:
            with tr.span("agent", "parent") as parent:
                with tr.span("tool", "child") as child:
                    pass
        assert parent.parent_id is None
        assert child.parent_id == parent.span_id

    def test_sibling_spans_share_a_parent(self, tr):
        with tr.trace("request") as trace:
            with tr.span("agent", "parent") as parent:
                with tr.span("tool", "a") as a:
                    pass
                with tr.span("tool", "b") as b:
                    pass
        assert a.parent_id == b.parent_id == parent.span_id

    def test_span_records_duration_and_ok_status(self, tr):
        with tr.trace("r"):
            with tr.span("tool", "t") as sp:
                pass
        assert sp.status == "ok"
        assert sp.duration_ms >= 0

    def test_exception_marks_span_as_error_and_propagates(self, tr):
        with tr.trace("r") as trace:
            with pytest.raises(ValueError):
                with tr.span("tool", "boom"):
                    raise ValueError("kaboom")
        sp = trace.spans[0]
        assert sp.status == "error"
        assert "ValueError: kaboom" in sp.error

    def test_by_kind_filters(self, tr):
        with tr.trace("r") as trace:
            with tr.span("agent", "a"):
                with tr.span("llm", "turn1"):
                    pass
                with tr.span("tool", "t1"):
                    pass
                with tr.span("tool", "t2"):
                    pass
        assert len(trace.by_kind("agent")) == 1
        assert len(trace.by_kind("llm")) == 1
        assert len(trace.by_kind("tool")) == 2

    def test_span_outside_a_trace_does_not_crash(self, tr):
        # Library code must be callable without an active trace.
        with tr.span("tool", "orphan") as sp:
            assert sp.span_id == "detached"

    def test_disabled_tracer_is_inert(self):
        off = Tracer(enabled=False, auto_save=False)
        with off.trace("r") as trace:
            with off.span("tool", "t") as sp:
                assert sp.span_id == "detached"
        assert trace.spans == []

    def test_waterfall_renders_every_span(self, tr):
        with tr.trace("r") as trace:
            with tr.span("agent", "researcher"):
                with tr.span("tool", "search_corpus"):
                    pass
        out = trace.waterfall()
        assert "researcher" in out and "search_corpus" in out
        # Child is indented further than its parent.
        lines = out.splitlines()
        assert lines[1].index("search_corpus") > lines[0].index("researcher")

    def test_summary_reports_counts(self, tr):
        with tr.trace("r") as trace:
            with tr.span("agent", "a"):
                with tr.span("tool", "web_search"):
                    pass
                with tr.span("tool", "web_search"):
                    pass
        summary = trace.summary()
        assert "agents        1" in summary
        assert "web_search×2" in summary

    def test_nested_tool_spans_are_counted_once(self, tr):
        # The dispatcher and the MCP tool each open a span for one logical call.
        with tr.trace("r") as trace:
            with tr.span("agent", "a"):
                with tr.span("tool", "dispatch:web_search"):
                    with tr.span("tool", "web_search"):
                        pass
        assert len(trace.by_kind("tool")) == 2
        assert len(trace.top_level_tools()) == 1
        assert "web_search×1" in trace.summary()

    def test_events_attach_to_current_span(self, tr):
        with tr.trace("r") as trace:
            with tr.span("tool", "t"):
                tr.event("something happened", detail="x")
        assert trace.spans[0].events[0]["message"] == "something happened"
        assert trace.spans[0].events[0]["detail"] == "x"

    def test_annotate_sets_attributes(self, tr):
        with tr.trace("r") as trace:
            with tr.span("tool", "t"):
                tr.annotate(rows=7)
        assert trace.spans[0].attributes["rows"] == 7


# ---------------------------------------------------------------------------
# 2. Token & cost tracking
# ---------------------------------------------------------------------------

class TestCostTracking:

    @pytest.fixture
    def hosted(self, monkeypatch):
        """Pin the hosted backend — the default is local, which is free."""
        monkeypatch.setenv("LLM_BACKEND", "openai")

    def test_local_inference_is_free(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "ollama")
        assert estimate_cost("llama3.1:8b", 1_000_000, 1_000_000) == 0.0

    def test_estimate_cost_matches_published_rates(self, hosted):
        # gpt-4o-mini: $0.15/1M in, $0.60/1M out
        assert estimate_cost("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)
        assert estimate_cost("gpt-4o-mini", 0, 1_000_000) == pytest.approx(0.60)

    def test_embedding_has_no_output_cost(self, hosted):
        assert estimate_cost("text-embedding-3-small", 0, 1_000_000) == 0.0

    def test_unknown_model_falls_back(self, hosted):
        assert estimate_cost("some-future-model", 1_000_000, 0) == pytest.approx(0.15)

    def test_costs_roll_up_to_the_trace(self, tr, fake_usage):
        with tr.trace("r") as trace:
            for _ in range(3):
                with tr.span("llm", "turn") as sp:
                    tr.record_llm_call(
                        sp, model="gpt-4o-mini",
                        usage=fake_usage(1000, 500), cost_usd=0.0005,
                    )
        assert trace.total_tokens == 4500
        assert trace.total_cost_usd == pytest.approx(0.0015)

    def test_record_llm_call_sets_token_fields(self, tr, fake_usage):
        with tr.trace("r"):
            with tr.span("llm", "turn") as sp:
                tr.record_llm_call(
                    sp, model="gpt-4o-mini", usage=fake_usage(120, 30), cost_usd=0.01
                )
        assert sp.prompt_tokens == 120
        assert sp.completion_tokens == 30
        assert sp.total_tokens == 150
        assert sp.attributes["model"] == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# 3. Context debugging
# ---------------------------------------------------------------------------

class TestContextDebugging:

    def test_prompt_window_is_captured(self, tr):
        messages = [
            {"role": "system", "content": "You are a researcher."},
            {"role": "user", "content": "What is Article 5?"},
        ]
        with tr.trace("r"):
            with tr.span("llm", "turn") as sp:
                tr.record_llm_call(sp, model="gpt-4o-mini", messages=messages)
        ctx = sp.attributes["context"]
        assert [m["role"] for m in ctx] == ["system", "user"]
        assert "Article 5" in ctx[1]["content"]

    def test_captured_context_is_redacted(self, tr):
        messages = [{"role": "user", "content": "key sk-abcdefghijklmnopqrstuvwxyz01"}]
        with tr.trace("r"):
            with tr.span("llm", "turn") as sp:
                tr.record_llm_call(sp, model="gpt-4o-mini", messages=messages)
        captured = sp.attributes["context"][0]["content"]
        assert "sk-abcdefghijklmnopqrstuvwxyz01" not in captured
        assert "[REDACTED:openai_key]" in captured

    def test_long_content_is_truncated(self, tr):
        messages = [{"role": "user", "content": "x" * 50_000}]
        with tr.trace("r"):
            with tr.span("llm", "turn") as sp:
                tr.record_llm_call(sp, model="gpt-4o-mini", messages=messages)
        captured = sp.attributes["context"][0]["content"]
        assert len(captured) < 10_000
        assert "chars]" in captured

    def test_retrieval_capture_records_scores_and_sources(self, tr):
        results = [
            {"source": "a.pdf", "page": 1, "score": 0.91, "text": "alpha passage"},
            {"source": "b.pdf", "page": 4, "score": 0.42, "text": "beta passage"},
        ]
        with tr.trace("r"):
            with tr.span("retrieval", "search") as sp:
                tr.record_retrieval(sp, query="what is x", results=results)
        assert sp.attributes["n_results"] == 2
        assert sp.attributes["top_score"] == 0.91
        assert sp.attributes["mean_score"] == pytest.approx(0.665)
        assert sp.attributes["retrieved"][0]["source"] == "a.pdf"
        assert "alpha passage" in sp.attributes["retrieved"][0]["preview"]

    def test_retrieval_capture_handles_no_results(self, tr):
        with tr.trace("r"):
            with tr.span("retrieval", "search") as sp:
                tr.record_retrieval(sp, query="q", results=[])
        assert sp.attributes["n_results"] == 0
        assert "top_score" not in sp.attributes


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSerialisation:

    def test_trace_round_trips_through_json(self, tr, fake_usage):
        with tr.trace("request", query="q") as trace:
            with tr.span("agent", "a"):
                with tr.span("llm", "turn") as sp:
                    tr.record_llm_call(
                        sp, model="gpt-4o-mini", usage=fake_usage(10, 5), cost_usd=0.001
                    )
        payload = json.loads(json.dumps(trace.to_dict()))
        assert payload["label"] == "request"
        assert payload["metadata"]["query"] == "q"
        assert payload["total_tokens"] == 15
        assert len(payload["spans"]) == 2

    def test_save_writes_a_readable_file(self, tr, tmp_path):
        with tr.trace("request") as trace:
            with tr.span("tool", "t"):
                pass
        path = trace.save(tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["trace_id"] == trace.trace_id
        assert data["spans"][0]["name"] == "t"

    def test_auto_save_writes_on_exit(self, tmp_path):
        auto = Tracer(enabled=True, auto_save=True)
        from app import observability

        observability.TRACE_DIR = tmp_path / "traces"
        with auto.trace("request") as trace:
            with auto.span("tool", "t"):
                pass
        assert (tmp_path / "traces" / f"{trace.trace_id}.json").exists()
