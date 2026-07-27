from __future__ import annotations

"""
scripts/cost_function.py

Tracks per-call API costs and persists a full audit log to disk.

Schema of cost_tracker.json:
{
  "total_usd": 0.012345,
  "history": [
    {
      "timestamp":          "2026-04-14T10:30:00+00:00",
      "user":               "prernamitra",
      "type":               "embedding" | "chat",
      "model":              "text-embedding-3-small" | "gpt-4o-mini",
      "prompt_tokens":      512,
      "completion_tokens":  0,
      "total_tokens":       512,
      "cost_usd":           0.000010
    },
    ...
  ]
}
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

COST_FILE = Path(os.environ.get("COST_FILE", "cost_tracker.json"))
BUDGET_USD = 5.00

# Pricing (USD per token)
_EMBEDDING_PRICE_PER_TOKEN = 0.02 / 1_000_000        # text-embedding-3-small
_CHAT_INPUT_PRICE_PER_TOKEN = 0.15 / 1_000_000       # gpt-4o-mini input
_CHAT_OUTPUT_PRICE_PER_TOKEN = 0.60 / 1_000_000      # gpt-4o-mini output


def _load() -> dict:
    if COST_FILE.exists():
        with open(COST_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    return {"total_usd": 0.0, "history": []}


def _save(data: dict) -> None:
    with open(COST_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def track_cost(
    response,
    call_type: str,
    user: str | None = None,
) -> float:
    """
    Record one API call and append it to the cost log.

    Parameters
    ----------
    response  : OpenAI response object (must have .usage)
    call_type : "embedding" or "chat"
    user      : identifier for who triggered the call;
                falls back to the COST_USER env var, then the OS username.

    Returns
    -------
    float — cost of this call in USD
    """
    if user is None:
        user = os.environ.get("COST_USER") or os.environ.get("USER", "unknown")

    usage = response.usage
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)

    # Local inference is free. Tokens are still recorded — they are the useful
    # signal for latency and context-window pressure — but billing them at
    # OpenAI rates would make the ledger fiction.
    from app import llm as _llm

    local = _llm.is_ollama()

    if call_type == "embedding":
        model = _llm.embed_model()
        cost = 0.0 if local else total_tokens * _EMBEDDING_PRICE_PER_TOKEN
    else:
        model = _llm.chat_model()
        cost = 0.0 if local else (
            prompt_tokens * _CHAT_INPUT_PRICE_PER_TOKEN
            + completion_tokens * _CHAT_OUTPUT_PRICE_PER_TOKEN
        )

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user": user,
        "type": call_type,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": round(cost, 8),
        "backend": _llm.backend_name(),
    }

    data = _load()
    data["total_usd"] = round(data["total_usd"] + cost, 8)
    data["budget_usd"] = BUDGET_USD
    data["remaining_usd"] = round(max(0.0, BUDGET_USD - data["total_usd"]), 4)
    data["percent_used"] = round((data["total_usd"] / BUDGET_USD) * 100, 2)
    data["history"].append(record)
    _save(data)

    remaining = max(0.0, BUDGET_USD - data["total_usd"])
    pct_used = (data["total_usd"] / BUDGET_USD) * 100

    print(
        f"[{record['timestamp']}]  user={user}  type={call_type}"
        f"  tokens={total_tokens}  cost=${cost:.6f}"
        f"  |  total=${data['total_usd']:.4f} / ${BUDGET_USD:.2f}"
        f"  ({pct_used:.1f}% used, ${remaining:.4f} remaining)"
    )
    return cost


def get_summary() -> dict:
    """Return total spend, per-user breakdown, and remaining budget."""
    data = _load()
    per_user: dict[str, float] = {}
    for entry in data.get("history", []):
        u = entry.get("user", "unknown")
        per_user[u] = round(per_user.get(u, 0.0) + entry.get("cost_usd", 0.0), 8)

    total = data.get("total_usd", 0.0)
    remaining = max(0.0, BUDGET_USD - total)

    return {
        "total_usd": total,
        "budget_usd": BUDGET_USD,
        "remaining_usd": round(remaining, 4),
        "percent_used": round((total / BUDGET_USD) * 100, 2),
        "per_user": per_user,
    }
