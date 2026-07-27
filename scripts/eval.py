from __future__ import annotations

"""
scripts/eval.py

Full evaluation runner with scorecard, LLM-as-judge, and regression tracking.

For each question:
  - Runs retrieval and records content hit + source hit
  - Generates an answer
  - Uses the active local model as judge (skip with --no-judge)
  - Classifies failure type: retrieval_miss or generation_error

Scorecard output:
  - Overall retrieval hit rate and generation pass rate
  - Breakdown by category and difficulty
  - Failure list with type and reason

Output: data/eval_results.json (compatible with ragas_eval.py)

Usage:
    uv run python -m scripts.eval
    uv run python -m scripts.eval --questions data/questions_1.json
    uv run python -m scripts.eval --no-judge
    uv run python -m scripts.eval --output data/eval_results_v2.json
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import llm as _llm
from scripts.chunk import (
    generate_answer,
    load_artifacts,
    retrieve,
    _get_openai_client,
)
from scripts.cost_function import get_summary

DATA_DIR  = Path(os.environ.get("PDF_OUTPUT_DIR", "data"))
EVAL_USER = "eval-runner"


# ---------------------------------------------------------------------------
# Expected-chunk helpers — same logic as retrieval_check.py
# ---------------------------------------------------------------------------

def _expected_pairs(q: dict) -> list[tuple[str, int]]:
    """Return (source, page) pairs for this question, supporting both file formats."""
    pairs: list[tuple[str, int]] = []
    for chunk in q.get("expected_chunks") or []:
        src, pg = chunk.get("source"), chunk.get("page")
        if src and pg is not None:
            pairs.append((src, int(pg)))
    if not pairs:
        for src_key, pg_key in [
            ("source", "page"),
            ("source_primary", "page_primary"),
            ("source_secondary", "page_secondary"),
        ]:
            src = q.get(src_key)
            pg  = q.get(pg_key)
            if src and pg is not None and isinstance(pg, int):
                pairs.append((src, pg))
    return pairs


def _content_hit(retrieved: list[dict], expected_answer: str) -> bool:
    if not expected_answer:
        return False
    phrases: list[str] = []
    for sentence in re.split(r"[.!?]", expected_answer):
        for clause in sentence.split(","):
            phrase = clause.strip().lower()
            if len(phrase) >= 6:
                phrases.append(phrase)
    combined = " ".join(c["text"].lower() for c in retrieved)
    return any(phrase in combined for phrase in phrases)


def _is_na(q: dict) -> bool:
    cat = q.get("category", "")
    sub = q.get("sub_type", "")
    return cat in ("out_of_scope", "ambiguous") or sub in (
        "out_of_scope", "ambiguous", "on_topic_unanswerable"
    )


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are an evaluator assessing whether a generated answer correctly addresses "
    "a question, given the expected answer as a reference.\n"
    "Respond with exactly PASS or FAIL on the first line, then one sentence explaining why.\n"
    "PASS means the generated answer captures the key facts from the expected answer.\n"
    "FAIL means the generated answer is wrong, incomplete, or off-topic."
)


def llm_judge(question: str, expected: str, generated: str) -> tuple[str, str]:
    """Return (verdict, reason). verdict is 'PASS' or 'FAIL'."""
    client = _get_openai_client()
    prompt = (
        f"Question: {question}\n\n"
        f"Expected answer: {expected}\n\n"
        f"Generated answer: {generated}"
    )
    response = client.chat.completions.create(
        model=_llm.chat_model(),
        temperature=0,
        max_tokens=120,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
    )
    text    = response.choices[0].message.content.strip()
    lines   = text.splitlines()
    verdict = "PASS" if lines[0].upper().startswith("PASS") else "FAIL"
    reason  = lines[1].strip() if len(lines) > 1 else ""
    return verdict, reason


# ---------------------------------------------------------------------------
# Scorecard printer
# ---------------------------------------------------------------------------

def print_scorecard(results: list[dict], use_judge: bool) -> None:
    applicable  = [r for r in results if not r["na"]]
    na_count    = len(results) - len(applicable)
    total       = len(applicable)

    if not total:
        print("No applicable questions (all N/A).")
        return

    ret_hits  = sum(1 for r in applicable if r["retrieval_content_hit"])
    gen_pass  = sum(1 for r in applicable if r.get("judge_verdict") == "PASS")

    print()
    print("=" * 64)
    print("EVAL SCORECARD")
    print("=" * 64)
    print(f"  Total questions:          {len(results)}")
    print(f"  Applicable (scored):      {total}")
    print(f"  N/A (out_of_scope etc):   {na_count}")
    print()
    print(f"  Retrieval content hit:    {ret_hits}/{total}  ({ret_hits/total:.1%})")
    if use_judge:
        print(f"  Generation pass rate:     {gen_pass}/{total}  ({gen_pass/total:.1%})")

    # By category
    cats: dict[str, dict] = {}
    for r in applicable:
        cat = r.get("category") or r.get("sub_type") or "uncategorised"
        if cat not in cats:
            cats[cat] = {"total": 0, "ret": 0, "gen": 0}
        cats[cat]["total"] += 1
        if r["retrieval_content_hit"]:
            cats[cat]["ret"] += 1
        if r.get("judge_verdict") == "PASS":
            cats[cat]["gen"] += 1

    print()
    print("  By category:")
    for cat, s in sorted(cats.items()):
        n       = s["total"]
        ret_pct = f"{s['ret']/n:.0%}" if n else "—"
        gen_pct = f"{s['gen']/n:.0%}" if (n and use_judge) else "—"
        print(f"    {cat:<22}  n={n}  ret={ret_pct}  gen={gen_pct}")

    # By difficulty
    diffs: dict[str, dict] = {}
    for r in applicable:
        diff = r.get("difficulty") or "unspecified"
        if diff not in diffs:
            diffs[diff] = {"total": 0, "ret": 0, "gen": 0}
        diffs[diff]["total"] += 1
        if r["retrieval_content_hit"]:
            diffs[diff]["ret"] += 1
        if r.get("judge_verdict") == "PASS":
            diffs[diff]["gen"] += 1

    print()
    print("  By difficulty:")
    for diff, s in sorted(diffs.items()):
        n       = s["total"]
        ret_pct = f"{s['ret']/n:.0%}" if n else "—"
        gen_pct = f"{s['gen']/n:.0%}" if (n and use_judge) else "—"
        print(f"    {diff:<12}  n={n}  ret={ret_pct}  gen={gen_pct}")

    # Failures
    failures = [
        r for r in applicable
        if not r["retrieval_content_hit"] or r.get("judge_verdict") == "FAIL"
    ]
    if failures:
        print()
        print(f"  Failed questions ({len(failures)}):")
        for r in failures:
            ftype   = r.get("failure_type", "unknown")
            qid     = r.get("id", "?")
            q_prev  = r["question"][:52]
            reason  = r.get("judge_reason", "")
            reason_str = f"  ← {reason}" if reason else ""
            print(f"    [{qid}] [{ftype}]  {q_prev}{reason_str}")

    print("=" * 64)
    print()


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_eval(
    questions_path: Path,
    output_path: Path,
    use_judge: bool = True,
) -> None:
    with open(questions_path, encoding="utf-8") as fh:
        questions = json.load(fh)

    chunks, index, bm25 = load_artifacts(DATA_DIR)

    results: list[dict] = []
    total = len(questions)

    for i, q in enumerate(questions, 1):
        question   = q["question"]
        expected   = q.get("expected_answer") or ""
        category   = q.get("category", "")
        difficulty = q.get("difficulty", "")
        sub_type   = q.get("sub_type", "")
        qid        = q.get("id", f"q{i:02d}")
        na         = _is_na(q)

        label = category or sub_type or "uncategorised"
        print(f"[{i}/{total}] {qid}  {label}")

        retrieved = retrieve(question, index, chunks, user=EVAL_USER, bm25=bm25)
        answer    = generate_answer(question, retrieved, user=EVAL_USER)

        exp             = _expected_pairs(q)
        retrieved_pairs = {(r["source"], r["page"]) for r in retrieved}

        ret_content = _content_hit(retrieved, expected) if expected else None
        ret_source  = (all(p in retrieved_pairs for p in exp) if exp else None)

        verdict = reason = None
        if use_judge and expected and not na:
            verdict, reason = llm_judge(question, expected, answer)

        # Classify failure type
        failure_type = None
        if not na and expected:
            if not ret_content:
                failure_type = "retrieval_miss"
            elif verdict == "FAIL":
                failure_type = "generation_error"

        record = {
            "id":                    qid,
            "question":              question,
            "expected_answer":       expected,
            "answer":                answer,
            "category":              category,
            "difficulty":            difficulty,
            "sub_type":              sub_type,
            "na":                    na,
            "retrieval_content_hit": bool(ret_content),
            "retrieval_source_hit":  ret_source,
            "judge_verdict":         verdict,
            "judge_reason":          reason,
            "failure_type":          failure_type,
            # Fields kept for ragas_eval.py compatibility
            "contexts":              [r["text"] for r in retrieved],
            "retrieved_sources": [
                {"source": r["source"], "page": r["page"], "score": r["score"]}
                for r in retrieved
            ],
        }
        results.append(record)

        status     = "N/A" if na else (" HIT" if ret_content else "MISS")
        judge_str  = f"  judge={verdict}" if verdict else ""
        print(f"  ret={status}{judge_str}")
        print(f"  A: {answer[:110]}{'…' if len(answer) > 110 else ''}")
        print()

    print_scorecard(results, use_judge)

    cost       = get_summary()
    applicable = [r for r in results if not r["na"]]
    failures   = [
        {
            "id":           r["id"],
            "question":     r["question"],
            "category":     r.get("category", ""),
            "difficulty":   r.get("difficulty", ""),
            "failure_type": r.get("failure_type", ""),
            "judge_reason": r.get("judge_reason", ""),
        }
        for r in applicable
        if not r["retrieval_content_hit"] or r.get("judge_verdict") == "FAIL"
    ]

    output = {
        "run_info": {
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "questions_file": str(questions_path),
            "total":          total,
            "cost_usd":       cost["total_usd"],
            "remaining_usd":  cost["remaining_usd"],
        },
        "summary": {
            "retrieval_hit_rate": (
                sum(1 for r in applicable if r["retrieval_content_hit"]) / len(applicable)
                if applicable else 0.0
            ),
            "generation_pass_rate": (
                sum(1 for r in applicable if r.get("judge_verdict") == "PASS") / len(applicable)
                if applicable else 0.0
            ),
            "failures": failures,
        },
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    print(f"Results saved → {output_path}")
    print("Run ragas_eval.py to compute faithfulness / precision / recall scores.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eval runner — full scorecard with retrieval hits and LLM-as-judge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python -m scripts.eval\n"
            "  uv run python -m scripts.eval --questions data/questions_1.json\n"
            "  uv run python -m scripts.eval --no-judge\n"
            "  uv run python -m scripts.eval --output data/eval_results_v2.json"
        ),
    )
    parser.add_argument(
        "--questions",
        default=str(DATA_DIR / "questions.json"),
        metavar="PATH",
        help="Path to questions file (default: data/questions.json)",
    )
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "eval_results.json"),
        metavar="PATH",
        help="Output path for results (default: data/eval_results.json)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM-as-judge step — retrieval stats only (saves cost)",
    )
    args = parser.parse_args()

    questions_path = Path(args.questions)
    output_path    = Path(args.output)

    if not questions_path.exists():
        print(f"ERROR: questions file not found at {questions_path}")
        sys.exit(1)
    if not (DATA_DIR / "my_index.faiss").exists():
        print("ERROR: FAISS index not found — run chunk.py first")
        sys.exit(1)

    run_eval(questions_path, output_path, use_judge=not args.no_judge)


if __name__ == "__main__":
    main()
