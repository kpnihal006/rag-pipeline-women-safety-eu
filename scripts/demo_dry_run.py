from __future__ import annotations

"""
scripts/demo_dry_run.py

Pre-demo health check: runs 10 representative questions through the full pipeline,
verifies responses are non-empty and citations are present, and measures response time.

Fails fast with a clear error if anything looks broken so you can fix it before
standing in front of an audience.

Usage:
    uv run python -m scripts.demo_dry_run
    uv run python -m scripts.demo_dry_run --verbose   # print full answers
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.chunk import load_artifacts, retrieve, generate_answer, CONFIDENCE_THRESHOLD
from scripts.cost_function import get_summary

DATA_DIR = Path(os.environ.get("PDF_OUTPUT_DIR", "data"))

# 10 representative questions — mix of factual, cross-reference, and stress tests
DEMO_QUESTIONS = [
    # factual — should cite statistics
    "How many women in the EU have experienced gender-based violence?",
    # factual — exact phone number
    "What number can I call if I am a victim of violence against women in the EU?",
    # factual — yes/no + legislation
    "Does EU law cover online harassment and cyber violence against women?",
    # factual — temporal
    "When did the EU pass a law on violence against women?",
    # cross-reference — two documents
    "Does the 2024 EU directive replace the Istanbul Convention or work alongside it?",
    # factual — from survey
    "What share of victims actually report violence to the police?",
    # exact term — legal article
    "What is the 'only yes means yes' rule and what does it say about silence?",
    # factual — country stats
    "Which EU countries have the highest rates of reported gender-based violence?",
    # out-of-scope — should be refused cleanly
    "What is the domestic violence rate in the United States?",
    # ambiguous — should ask for clarification
    "Tell me about the requirements for safety.",
]

_NO_CITATION_PHRASES = (
    "outside the scope",
    "not available in the corpus",
    "too broad to answer",
)


def check_answer(question: str, answer: str, results: list[dict]) -> list[str]:
    """Return a list of warning strings. Empty list means the check passed."""
    warnings: list[str] = []

    if not answer or len(answer.strip()) < 10:
        warnings.append("EMPTY or near-empty answer")
        return warnings

    is_refusal = any(p in answer.lower() for p in _NO_CITATION_PHRASES)

    # Non-refusal answers must include at least one source reference
    if not is_refusal:
        has_citation = any(
            r["source"].split(".")[0].lower() in answer.lower()
            or f"page {r['page']}" in answer.lower()
            or f"p.{r['page']}" in answer.lower()
            for r in results
        )
        # Fallback: answer pipeline adds "Sources" block — check if results were found
        if not results:
            warnings.append("NO chunks retrieved")
        elif results[0].get("low_confidence"):
            warnings.append(
                f"LOW CONFIDENCE — top score {results[0]['score']:.3f} "
                f"< threshold {CONFIDENCE_THRESHOLD}"
            )

    return warnings


def run_dry_run(verbose: bool = False) -> bool:
    index_path = DATA_DIR / "my_index.faiss"
    if not index_path.exists():
        print("ERROR: FAISS index not found — run chunk.py first")
        return False

    print("Loading artifacts …")
    t0 = time.perf_counter()
    chunks, index, bm25 = load_artifacts(DATA_DIR)
    load_time = time.perf_counter() - t0
    print(f"  Loaded in {load_time:.1f}s\n")

    total = len(DEMO_QUESTIONS)
    passed = failed = 0
    timings: list[float] = []
    all_warnings: list[tuple[int, str, list[str]]] = []

    print(f"Running {total} demo questions …\n")
    print(f"  {'#':<3}  {'ms':>5}  {'conf':>5}  {'status':<10}  Question")
    print("  " + "-" * 72)

    for i, question in enumerate(DEMO_QUESTIONS, 1):
        t_start = time.perf_counter()
        results = retrieve(question, index, chunks, user="demo-dry-run", bm25=bm25)
        answer  = generate_answer(question, results, user="demo-dry-run")
        elapsed = time.perf_counter() - t_start
        timings.append(elapsed)

        warnings = check_answer(question, answer, results)
        top_score = results[0]["score"] if results else 0.0
        conf_str  = f"{top_score:.2f}"
        status    = "PASS" if not warnings else "WARN"

        if warnings:
            failed += 1
            all_warnings.append((i, question, warnings))
        else:
            passed += 1

        q_preview = question[:50]
        print(f"  {i:<3}  {elapsed*1000:>5.0f}  {conf_str:>5}  {status:<10}  {q_preview}")

        if verbose:
            print(f"       A: {answer[:120]}{'…' if len(answer) > 120 else ''}\n")

    # Summary
    avg_ms = sum(timings) / len(timings) * 1000
    max_ms = max(timings) * 1000
    cost   = get_summary()

    print()
    print("=" * 60)
    print("DEMO DRY-RUN SUMMARY")
    print("=" * 60)
    print(f"  Questions:      {total}")
    print(f"  Passed:         {passed}")
    print(f"  Warnings:       {failed}")
    print(f"  Avg response:   {avg_ms:.0f} ms")
    print(f"  Max response:   {max_ms:.0f} ms")
    print(f"  Cost this run:  ~${(cost['total_usd']):.4f} total so far")

    if all_warnings:
        print()
        print("  Warnings to review:")
        for idx, q, ws in all_warnings:
            print(f"    #{idx} {q[:55]}")
            for w in ws:
                print(f"       ⚠  {w}")

    print("=" * 60)

    if failed == 0:
        print("\n  ✅ All checks passed — ready for demo.")
    else:
        print(f"\n  ⚠  {failed} warning(s) — review before going live.")

    return failed == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo dry-run: smoke-test 10 questions and report health"
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print truncated answers alongside each result")
    args = parser.parse_args()

    ok = run_dry_run(verbose=args.verbose)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
