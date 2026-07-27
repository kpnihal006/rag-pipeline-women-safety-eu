from __future__ import annotations

"""
scripts/retrieval_quality.py

Measure retrieval with three independent metrics and report their agreement.

WHY THIS EXISTS
---------------
The headline "retrieval hit rate" in `scripts/eval.py` is computed by checking
whether a >=6-character clause of the *reference answer* appears verbatim in the
retrieved text. Reference answers are human paraphrases of the source, not
quotations, so correct retrieval routinely scores as a miss — question q01 is
labelled `retrieval_miss` while the judge simultaneously rates its answer
correct.

Replacing it with page-level gold does not settle the matter either: the two
metrics disagree on 9 of 15 questions, which is close to no agreement at all.
When two proxies for the same quantity disagree that often, neither is
measuring it.

This script therefore adds the metric that actually states the question —
**context sufficiency**: given the retrieved passages and the reference answer,
could the question be answered from these passages alone? That is judged by the
local model, which is slower than a substring check but is measuring the right
thing rather than a lexical shadow of it.

All three are reported side by side, with pairwise agreement, so the report can
state how much confidence the retrieval numbers deserve.

Usage:
    uv run python -m scripts.retrieval_quality
    uv run python -m scripts.retrieval_quality --k 8 --limit 5
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import llm as _llm  # noqa: E402
from scripts.chunk import load_artifacts, retrieve  # noqa: E402

DATA_DIR = Path(os.environ.get("PDF_OUTPUT_DIR", "data"))

_SUFFICIENCY_SYSTEM = """\
You decide whether a set of passages is sufficient to answer a question.

You are given a QUESTION, the REFERENCE ANSWER, and the retrieved PASSAGES.

Reply with exactly one word:
  SUFFICIENT   — the passages contain the information in the reference answer,
                 even if worded differently
  INSUFFICIENT — the passages do not contain it

Judge only whether the information is present. Do not judge writing quality.
"""


def substring_hit(passages: list[dict], expected: str) -> bool:
    """The incumbent metric: a clause of the reference answer appears verbatim."""
    if not expected:
        return False
    phrases = []
    for sentence in re.split(r"[.!?]", expected):
        for clause in sentence.split(","):
            phrase = clause.strip().lower()
            if len(phrase) >= 6:
                phrases.append(phrase)
    combined = " ".join(p["text"].lower() for p in passages)
    return any(p in combined for p in phrases)


def page_hit(passages: list[dict], item: dict) -> bool | None:
    """Gold pages declared by the question set."""
    gold = {(str(e["source"]).lower(), int(e["page"]))
            for e in (item.get("expected_chunks") or [])
            if e.get("source") and e.get("page") is not None}
    if not gold:
        return None
    for p in passages:
        key = (str(p["source"]).lower(), int(p["page"]))
        for g in gold:
            if key[1] == g[1] and (
                key[0] == g[0]
                or key[0].split(".")[0] in g[0]
                or g[0].split(".")[0] in key[0]
            ):
                return True
    return False


def sufficiency_hit(client, model: str, question: str, expected: str,
                    passages: list[dict]) -> bool:
    ctx = "\n\n".join(
        f"[{i}] {p['source']} p{p['page']}\n{p['text']}"
        for i, p in enumerate(passages, 1)
    )
    resp = client.chat.completions.create(
        model=model, temperature=0, max_tokens=8,
        messages=[
            {"role": "system", "content": _SUFFICIENCY_SYSTEM},
            {"role": "user", "content":
             f"QUESTION: {question}\n\nREFERENCE ANSWER: {expected}\n\n"
             f"PASSAGES:\n{ctx}"},
        ],
    )
    out = (resp.choices[0].message.content or "").upper()
    return "INSUFFICIENT" not in out and "SUFFICIENT" in out


def agreement(a: list[bool], b: list[bool]) -> float:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return float("nan")
    return sum(1 for x, y in pairs if x == y) / len(pairs)


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieval metric comparison")
    ap.add_argument("--questions", default=str(DATA_DIR / "questions.json"))
    ap.add_argument("--output", default=str(DATA_DIR / "retrieval_quality.json"))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    ok, detail = _llm.health_check()
    print(f"Backend: {detail}")
    if not ok:
        sys.exit("Backend unavailable.")

    items = [q for q in json.loads(Path(args.questions).read_text())
             if q.get("expected_answer")
             and q.get("sub_type") not in ("out_of_scope", "ambiguous",
                                           "on_topic_unanswerable")]
    if args.limit:
        items = items[: args.limit]

    chunks, index, bm25 = load_artifacts(DATA_DIR)
    client = _llm.get_client()
    model = _llm.chat_model()

    rows = []
    print(f"\n{'id':<7}{'type':<20}{'substring':>10}{'page':>7}{'sufficient':>12}")
    print("-" * 58)
    for n, it in enumerate(items, 1):
        passages = retrieve(it["question"], index, chunks, k=args.k, bm25=bm25,
                            expand=False)
        sub = substring_hit(passages, it["expected_answer"])
        pg = page_hit(passages, it)
        suf = sufficiency_hit(client, model, it["question"],
                              it["expected_answer"], passages)
        rows.append({
            "id": it.get("id", f"q{n:02d}"),
            "sub_type": it.get("sub_type", ""),
            "substring_hit": sub, "page_hit": pg, "sufficient": suf,
        })
        print(f"{rows[-1]['id']:<7}{rows[-1]['sub_type']:<20}"
              f"{str(sub):>10}{str(pg):>7}{str(suf):>12}", flush=True)

    n = len(rows)
    sub_r = sum(bool(r["substring_hit"]) for r in rows) / n
    pg_rows = [r for r in rows if r["page_hit"] is not None]
    pg_r = (sum(bool(r["page_hit"]) for r in pg_rows) / len(pg_rows)
            if pg_rows else float("nan"))
    suf_r = sum(bool(r["sufficient"]) for r in rows) / n

    print("\n" + "=" * 58)
    print("RETRIEVAL MEASURED THREE WAYS")
    print("=" * 58)
    print(f"  verbatim-substring hit   {sub_r:>7.1%}   (incumbent metric)")
    print(f"  page-level gold hit      {pg_r:>7.1%}")
    print(f"  context sufficiency      {suf_r:>7.1%}   (judged)")
    print("\n  Pairwise agreement:")
    print(f"    substring vs page        "
          f"{agreement([r['substring_hit'] for r in rows], [r['page_hit'] for r in rows]):.1%}")
    print(f"    substring vs sufficiency "
          f"{agreement([r['substring_hit'] for r in rows], [r['sufficient'] for r in rows]):.1%}")
    print(f"    page vs sufficiency      "
          f"{agreement([r['page_hit'] for r in rows], [r['sufficient'] for r in rows]):.1%}")
    print("=" * 58)
    print(f"  n = {n}, k = {args.k}")

    Path(args.output).write_text(json.dumps({
        "k": args.k, "n": n,
        "substring_hit_rate": round(sub_r, 4),
        "page_hit_rate": round(pg_r, 4),
        "sufficiency_rate": round(suf_r, 4),
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
