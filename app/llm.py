from __future__ import annotations

"""
app/llm.py

Single point of truth for which LLM backend the system talks to.

**Ollama is the default.** The pipeline runs fully locally — no API key, no
per-token cost, and no data leaving the machine, which matters for a system
that handles questions about personal safety. OpenAI remains available as an
opt-in backend for comparison runs and for the evaluation harness.

Selection order:

  1. `LLM_BACKEND=ollama|openai`        — explicit wins
  2. `USE_OLLAMA=false` / `USE_OPENAI=true` — legacy toggles
  3. default: **ollama**

Both backends are reached through the OpenAI Python client, because Ollama
serves an OpenAI-compatible `/v1` endpoint. That keeps one call path — including
tool/function calling — rather than two code paths that drift apart.

Environment:
    LLM_BACKEND         "ollama" (default) or "openai"
    OLLAMA_HOST         default http://localhost:11434
    OLLAMA_MODEL        default llama3.1:8b
    OLLAMA_EMBED_MODEL  default nomic-embed-text
    OPENAI_API_KEY      required only when the backend is openai
    OPENAI_MODEL        default gpt-4o-mini
    OPENAI_EMBED_MODEL  default text-embedding-3-small

Usage:
    from app.llm import get_client, chat_model, embed_model, backend_name

    client = get_client()
    resp = client.chat.completions.create(model=chat_model(), messages=[...])
"""

import logging
import os
from functools import lru_cache

import openai
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

OPENAI_CHAT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")

#: Embedding width. Ollama's nomic-embed-text is fixed at 768; OpenAI's
#: text-embedding-3-small is Matryoshka-truncatable and we use 1536.
OLLAMA_EMBED_DIM = 768
OPENAI_EMBED_DIM = 1536


def _truthy(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    return value.strip().lower() in ("1", "true", "yes", "on")


def backend_name() -> str:
    """Resolve the active backend. Ollama unless explicitly told otherwise."""
    explicit = os.environ.get("LLM_BACKEND", "").strip().lower()
    if explicit in ("ollama", "openai"):
        return explicit

    use_ollama = _truthy(os.environ.get("USE_OLLAMA"))
    if use_ollama is False:
        return "openai"
    if use_ollama is True:
        return "ollama"

    if _truthy(os.environ.get("USE_OPENAI")):
        return "openai"

    return "ollama"


def is_ollama() -> bool:
    return backend_name() == "ollama"


def chat_model() -> str:
    return OLLAMA_MODEL if is_ollama() else OPENAI_CHAT_MODEL


def embed_model() -> str:
    return OLLAMA_EMBED_MODEL if is_ollama() else OPENAI_EMBEDDING_MODEL


def embed_dim() -> int:
    return OLLAMA_EMBED_DIM if is_ollama() else OPENAI_EMBED_DIM


def supports_dimensions() -> bool:
    """Whether the embeddings endpoint honours the `dimensions` parameter.

    Ollama ignores it and returns the model's native width, so callers must not
    send it — passing it produces a confusing dimension mismatch against FAISS.
    """
    return not is_ollama()


@lru_cache(maxsize=4)
def _client_for(backend: str, host: str, key: str) -> openai.OpenAI:
    if backend == "ollama":
        # Ollama ignores the key but the client requires a non-empty string.
        return openai.OpenAI(base_url=f"{host.rstrip('/')}/v1", api_key="ollama")
    if not key:
        raise RuntimeError(
            "LLM_BACKEND=openai but OPENAI_API_KEY is not set. Either set the "
            "key or use the default local backend (LLM_BACKEND=ollama)."
        )
    return openai.OpenAI(api_key=key)


def get_client() -> openai.OpenAI:
    """An OpenAI-compatible client pointed at the active backend."""
    backend = backend_name()
    return _client_for(backend, OLLAMA_HOST, os.environ.get("OPENAI_API_KEY", ""))


def health_check() -> tuple[bool, str]:
    """Confirm the backend is reachable and the configured model is present."""
    backend = backend_name()
    if backend == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            return False, "OPENAI_API_KEY is not set"
        return True, f"openai · {OPENAI_CHAT_MODEL}"

    import requests

    try:
        resp = requests.get(f"{OLLAMA_HOST.rstrip('/')}/api/tags", timeout=5)
        resp.raise_for_status()
        tags = [m["name"] for m in resp.json().get("models", [])]
    except Exception as exc:
        return False, (
            f"Ollama unreachable at {OLLAMA_HOST} ({exc}). "
            "Start it with `ollama serve`, or set LLM_BACKEND=openai."
        )

    def present(name: str) -> bool:
        return any(t == name or t.split(":")[0] == name.split(":")[0] for t in tags)

    missing = [
        m for m in (OLLAMA_MODEL, OLLAMA_EMBED_MODEL) if not present(m)
    ]
    if missing:
        return False, (
            f"Ollama is running but these models are missing: {', '.join(missing)}. "
            f"Pull them with: {' && '.join(f'ollama pull {m}' for m in missing)}"
        )
    return True, f"ollama · {OLLAMA_MODEL} · {OLLAMA_EMBED_MODEL} @ {OLLAMA_HOST}"


def describe() -> str:
    ok, detail = health_check()
    return f"{'OK' if ok else 'UNAVAILABLE'} — {detail}"


__all__ = [
    "backend_name", "is_ollama", "chat_model", "embed_model", "embed_dim",
    "supports_dimensions", "get_client", "health_check", "describe",
    "OLLAMA_HOST", "OLLAMA_MODEL", "OLLAMA_EMBED_MODEL",
]
