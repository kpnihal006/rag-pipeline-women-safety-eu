from __future__ import annotations

"""Shared pytest fixtures.

The whole suite is offline: no OpenAI calls, no network, no FAISS artifacts
required. Anything that would reach outside the process is stubbed, so `pytest`
runs in CI on a clean checkout.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _clean_guard():
    """Reset security events between tests so counts are per-test."""
    from app.security import guard

    guard.reset()
    yield
    guard.reset()


@pytest.fixture(autouse=True)
def _clean_tracer(tmp_path, monkeypatch):
    """Point traces at a temp dir and clear state between tests."""
    from app import observability

    monkeypatch.setattr(observability, "TRACE_DIR", tmp_path / "traces")
    observability.tracer.reset()
    yield
    observability.tracer.reset()


@pytest.fixture
def guard():
    from app.security import guard as g

    return g


@pytest.fixture
def tracer():
    from app.observability import tracer as t

    return t


@pytest.fixture
def strict_guard():
    """A guard with PII redaction turned on, isolated from the global one."""
    from app.security import SecurityGuard

    return SecurityGuard(strict_pii=True)


class FakeUsage:
    def __init__(self, prompt_tokens: int = 100, completion_tokens: int = 50):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


@pytest.fixture
def fake_usage():
    return FakeUsage
