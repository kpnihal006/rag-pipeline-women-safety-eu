from __future__ import annotations

"""Tests for the RAG substrate: chunking, cost accounting, and text cleaning.

These cover the data-engineering layer at the heart of the
process — "shit in, shit out still prevails". Retrieval quality itself is
measured by scripts/eval.py and scripts/ragas_eval.py, not here.
"""

import json

import pytest

from scripts.chunk import CHUNK_OVERLAP, CHUNK_SIZE, _split
from scripts.extract import clean_text


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

class TestSplit:

    def test_short_text_stays_one_chunk(self):
        assert len(_split("A single short sentence about EU law.")) == 1

    def test_empty_text_produces_no_chunks(self):
        assert _split("") == []
        assert _split("     \n\n  ") == []

    def test_long_text_is_split_on_sentence_boundaries(self):
        text = " ".join(
            f"This is sentence number {i} about European Union policy."
            for i in range(200)
        )
        chunks = _split(text)
        assert len(chunks) > 1
        # Sentence-aware: no chunk ends mid-sentence.
        for chunk in chunks:
            assert chunk.rstrip().endswith(".")

    def test_a_single_long_sentence_is_not_cut(self):
        # The chunker is sentence-aware by design: it never cuts mid-sentence,
        # so one unpunctuated blob stays whole even past CHUNK_SIZE.
        assert len(_split("word " * 2000)) == 1

    def test_chunks_respect_the_size_budget(self):
        # Allow headroom: the splitter breaks on boundaries, not mid-word.
        for chunk in _split("Sentence about policy. " * 500):
            assert len(chunk) <= CHUNK_SIZE * 2

    def test_no_chunk_is_empty_or_whitespace(self):
        for chunk in _split("Some text. " * 300):
            assert chunk.strip()

    def test_content_is_not_lost(self):
        text = " ".join(f"token{i}" for i in range(400))
        joined = " ".join(_split(text))
        assert "token0" in joined
        assert "token399" in joined

    def test_overlap_is_configured_smaller_than_chunk_size(self):
        assert 0 < CHUNK_OVERLAP < CHUNK_SIZE


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

class TestCleanText:

    def test_returns_a_string(self):
        assert isinstance(clean_text("Some raw PDF text\n\npage 1"), str)

    def test_empty_input_is_safe(self):
        assert clean_text("") == ""

    def test_meaningful_content_survives(self):
        text = "Article 50 of the Istanbul Convention obliges parties to act."
        assert "Istanbul Convention" in clean_text(text)

    def test_whitespace_is_normalised(self):
        cleaned = clean_text("Line one\n\n\n\n\nLine two")
        assert "\n\n\n\n" not in cleaned


# ---------------------------------------------------------------------------
# Cost tracking ledger
# ---------------------------------------------------------------------------

class TestCostTracker:

    @pytest.fixture
    def ledger(self, tmp_path, monkeypatch):
        """Ledger pinned to the hosted backend, where calls actually cost money."""
        import scripts.cost_function as cf

        monkeypatch.setattr(cf, "COST_FILE", tmp_path / "cost.json")
        monkeypatch.setenv("LLM_BACKEND", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used-offline")
        return cf

    @pytest.fixture
    def local_ledger(self, tmp_path, monkeypatch):
        import scripts.cost_function as cf

        monkeypatch.setattr(cf, "COST_FILE", tmp_path / "cost.json")
        monkeypatch.setenv("LLM_BACKEND", "ollama")
        return cf

    class Usage:
        def __init__(self, p, c):
            self.prompt_tokens = p
            self.completion_tokens = c
            self.total_tokens = p + c

    class Resp:
        def __init__(self, usage):
            self.usage = usage

    def test_chat_cost_is_recorded(self, ledger):
        cost = ledger.track_cost(
            self.Resp(self.Usage(1000, 500)), call_type="chat", user="tester"
        )
        assert cost > 0
        data = json.loads(ledger.COST_FILE.read_text())
        assert data["total_usd"] == pytest.approx(cost)
        assert data["history"][0]["user"] == "tester"
        assert data["history"][0]["type"] == "chat"

    def test_costs_accumulate_across_calls(self, ledger):
        a = ledger.track_cost(self.Resp(self.Usage(100, 50)), call_type="chat")
        b = ledger.track_cost(self.Resp(self.Usage(200, 80)), call_type="chat")
        data = json.loads(ledger.COST_FILE.read_text())
        assert data["total_usd"] == pytest.approx(a + b)
        assert len(data["history"]) == 2

    def test_embedding_is_cheaper_than_chat_for_the_same_tokens(self, ledger):
        emb = ledger.track_cost(self.Resp(self.Usage(1000, 0)), call_type="embedding")
        chat = ledger.track_cost(self.Resp(self.Usage(1000, 0)), call_type="chat")
        assert emb < chat

    def test_local_backend_records_tokens_but_no_cost(self, local_ledger):
        # Tokens still matter for latency and context pressure; dollars do not.
        cost = local_ledger.track_cost(
            self.Resp(self.Usage(5000, 2000)), call_type="chat", user="tester"
        )
        assert cost == 0.0
        data = json.loads(local_ledger.COST_FILE.read_text())
        assert data["total_usd"] == 0.0
        assert data["history"][0]["total_tokens"] == 7000
        assert data["history"][0]["backend"] == "ollama"

    def test_summary_reports_budget_state(self, ledger):
        ledger.track_cost(self.Resp(self.Usage(1000, 500)), call_type="chat")
        summary = ledger.get_summary()
        assert summary["total_usd"] > 0
        assert ledger.BUDGET_USD == 5.00


class TestAnswerReferences:
    """Evidence references are a hard requirement, not a nice-to-have.

    Relying on the prompt alone produced references in only 55% of answers —
    a local 8B model forgets to cite roughly half the time. These tests pin the
    deterministic fallback that guarantees the requirement is met.
    """

    RESULTS = [
        {"source": "istanbul.pdf", "page": 12, "text": "..."},
        {"source": "victims_directive.pdf", "page": 4, "text": "..."},
    ]

    def test_uncited_claim_needs_sources(self):
        from scripts.chunk import _needs_sources

        assert _needs_sources("Police must respond promptly.", self.RESULTS)

    @pytest.mark.parametrize("answer", [
        "According to istanbul.pdf the police must respond.",
        "See page 12 for the requirement.",
        "The requirement is stated in passage [1].",
    ])
    def test_already_cited_is_left_alone(self, answer):
        from scripts.chunk import _needs_sources

        assert not _needs_sources(answer, self.RESULTS)

    @pytest.mark.parametrize("refusal", [
        "This specific information is not in the corpus.",
        "That question is outside the scope of this corpus.",
        "This question is too broad to answer precisely.",
    ])
    def test_refusals_get_no_sources_block(self, refusal):
        # A refusal cites nothing by design; appending sources would imply
        # evidence that was never used.
        from scripts.chunk import _needs_sources

        assert not _needs_sources(refusal, self.RESULTS)

    def test_empty_answer_and_no_results_are_skipped(self):
        from scripts.chunk import _needs_sources

        assert not _needs_sources("", self.RESULTS)
        assert not _needs_sources("A claim.", [])

    def test_existing_sources_block_is_not_duplicated(self):
        from scripts.chunk import _needs_sources

        assert not _needs_sources("Answer.\n\n---\n**Sources**\n- a.pdf — p.1",
                                  self.RESULTS)
