from __future__ import annotations

"""Tests for app/llm.py — backend selection and model resolution.

The default must be the local backend: a silent fallback to a paid hosted API
would be both a cost surprise and a privacy problem for a system handling
personal-safety questions. These tests pin that contract.
"""

import pytest

from app import llm


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for var in ("LLM_BACKEND", "USE_OLLAMA", "USE_OPENAI"):
        monkeypatch.delenv(var, raising=False)
    yield


class TestBackendSelection:

    def test_default_is_local(self):
        assert llm.backend_name() == "ollama"
        assert llm.is_ollama() is True

    def test_explicit_backend_wins(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "openai")
        assert llm.backend_name() == "openai"
        monkeypatch.setenv("LLM_BACKEND", "ollama")
        assert llm.backend_name() == "ollama"

    def test_explicit_backend_overrides_legacy_toggle(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "ollama")
        monkeypatch.setenv("USE_OLLAMA", "false")
        assert llm.backend_name() == "ollama"

    @pytest.mark.parametrize("value", ["false", "0", "no", "off"])
    def test_use_ollama_false_selects_hosted(self, monkeypatch, value):
        monkeypatch.setenv("USE_OLLAMA", value)
        assert llm.backend_name() == "openai"

    @pytest.mark.parametrize("value", ["true", "1", "yes", "on"])
    def test_use_ollama_true_selects_local(self, monkeypatch, value):
        monkeypatch.setenv("USE_OLLAMA", value)
        assert llm.backend_name() == "ollama"

    def test_use_openai_toggle(self, monkeypatch):
        monkeypatch.setenv("USE_OPENAI", "true")
        assert llm.backend_name() == "openai"

    def test_unrecognised_backend_falls_back_to_local(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "gemini")
        assert llm.backend_name() == "ollama"

    def test_empty_toggle_is_ignored(self, monkeypatch):
        monkeypatch.setenv("USE_OLLAMA", "")
        assert llm.backend_name() == "ollama"


class TestModelResolution:

    def test_local_models(self):
        assert llm.chat_model() == llm.OLLAMA_MODEL
        assert llm.embed_model() == llm.OLLAMA_EMBED_MODEL
        assert llm.embed_dim() == 768

    def test_hosted_models(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "openai")
        assert llm.chat_model() == llm.OPENAI_CHAT_MODEL
        assert llm.embed_dim() == 1536

    def test_dimensions_param_only_for_hosted(self, monkeypatch):
        # Ollama ignores `dimensions` and returns its native width; sending it
        # produces a confusing mismatch against FAISS rather than an error.
        assert llm.supports_dimensions() is False
        monkeypatch.setenv("LLM_BACKEND", "openai")
        assert llm.supports_dimensions() is True

    def test_local_and_hosted_dims_differ(self, monkeypatch):
        local = llm.embed_dim()
        monkeypatch.setenv("LLM_BACKEND", "openai")
        assert llm.embed_dim() != local


class TestClient:

    def test_local_client_needs_no_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = llm.get_client()
        assert "11434" in str(client.base_url) or "/v1" in str(client.base_url)

    def test_hosted_client_without_key_fails_loudly(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        llm._client_for.cache_clear()
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            llm.get_client()
        llm._client_for.cache_clear()


class TestArtifactScoping:

    def test_index_filenames_differ_by_backend(self, monkeypatch):
        from scripts.chunk import _chunks_filename, _index_filename

        local_i, local_c = _index_filename(), _chunks_filename()
        monkeypatch.setenv("LLM_BACKEND", "openai")
        # Distinct names prevent loading a 768-dim index with 1536-dim queries.
        assert _index_filename() != local_i
        assert _chunks_filename() != local_c
