from __future__ import annotations

"""
scripts/retrieval_check.py

Debugging tool: run retrieval for a question and inspect exactly what was found.

For each run shows:
  - Top-k retrieved chunks (text, source, page, similarity score)
  - ✅ / ❌ hit markers against expected chunks
  - Content hit  — answer phrase found anywhere in retrieved text
  - Source hit   — annotated (source, page) pair is in top-k

Works with both question file formats:
  - questions.json   (flat source/page fields, has "id")
  - questions_1.json (expected_chunks list, has "difficulty" / "sub_type")

Usage:
    uv run python -m scripts.retrieval_check --id q09
    uv run python -m scripts.retrieval_check --question "What is the Istanbul Convention?"
    uv run python -m scripts.retrieval_check --all
    uv run python -m scripts.retrieval_check --all --questions data/questions_1.json
    uv run python -m scripts.retrieval_check --id q09 --k 10
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

from scripts.chunk import load_artifacts, retrieve

DATA_DIR = Path(os.environ.get("PDF_OUTPUT_DIR", "data"))
RETRIEVAL_USER = "retrieval-check"
W = 72  # output width


# ---------------------------------------------------------------------------
# Question file helpers — support both formats
# ---------------------------------------------------------------------------

def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def build_index(questions: list[dict]) -> dict[str, dict]:
    """Index by id (if present) and by lower-cased question text."""
    idx: dict[str, dict] = {}
    for q in questions:
        qid = q.get("id")
        if qid:
            idx[qid] = q
        idx[q["question"].strip().lower()] = q
    return idx


# kept for external callers that import this name
load_question_index = build_index


def expected_pairs(q: dict) -> list[tuple[str, int]]:
    """Return (source, page) pairs that count as a retrieval hit.

    Handles both formats:
      questions.json   — flat source/page, source_primary/page_primary, source_secondary/page_secondary
      questions_1.json — expected_chunks: [{"source": ..., "page": ...}]
    """
    pairs: list[tuple[str, int]] = []

    # questions_1.json — list format takes priority
    for chunk in q.get("expected_chunks") or []:
        src = chunk.get("source")
        pg  = chunk.get("page")
        if src and pg is not None:
            pairs.append((src, int(pg)))

    # questions.json — flat fields (only if expected_chunks didn't fire)
    if not pairs:
        for src_key, pg_key in [
            ("source", "page"),
            ("source_primary", "page_primary"),
            ("source_secondary", "page_secondary"),
        ]:
            src = q.get(src_key)
            pg  = q.get(pg_key)
            # skip string page values like "5, 95" — no single annotated page
            if src and pg is not None and isinstance(pg, int):
                pairs.append((src, pg))

    return pairs


def answer_hit(retrieved: list[dict], expected_answer: str) -> bool:
    """Return True if any retrieved chunk contains a key phrase from the expected answer.

    Splits on sentence and comma boundaries; requires at least one 6+ char phrase match.
    """
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


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def _wrap(text: str, indent: int = 2) -> None:
    text = text.replace("\n", " ")
    cols = W - indent
    for start in range(0, len(text), cols):
        print(" " * indent + text[start : start + cols])


def print_results(
    question: str,
    retrieved: list[dict],
    q_meta: dict | None = None,
) -> None:
    print()
    print("=" * W)
    print(f"QUESTION  {question}")
    if q_meta:
        parts = [
            q_meta.get("category", ""),
            q_meta.get("difficulty", ""),
            q_meta.get("sub_type", ""),
        ]
        label = " / ".join(p for p in parts if p)
        if label:
            print(f"          [{label}]")
    print("=" * W)

    is_na = False
    expected: list[tuple[str, int]] = []
    if q_meta:
        cat = q_meta.get("category", "")
        sub = q_meta.get("sub_type", "")
        if cat in ("out_of_scope", "ambiguous") or sub in (
            "out_of_scope", "ambiguous", "on_topic_unanswerable"
        ):
            is_na = True
            print("  (no expected chunk — out_of_scope / ambiguous / on_topic_unanswerable)")
        else:
            expected = expected_pairs(q_meta)

    retrieved_pairs = [(r["source"], r["page"]) for r in retrieved]

    for rank, chunk in enumerate(retrieved, 1):
        src, pg, score = chunk["source"], chunk["page"], chunk["score"]
        marker = "  ✅ EXPECTED" if (src, pg) in expected else ""
        print(f"\n  #{rank}  {src}  p.{pg}  score={score:.4f}{marker}")
        print("  " + "-" * (W - 2))
        _wrap(chunk["text"])

    # Hit / miss summary
    if not is_na:
        exp_answer = (q_meta or {}).get("expected_answer", "")
        if expected or exp_answer:
            print()
            print("-" * W)

        if expected:
            retrieved_set = set(retrieved_pairs)
            for src, pg in expected:
                if (src, pg) in retrieved_set:
                    rank = retrieved_pairs.index((src, pg)) + 1
                    print(f"  ✅ SOURCE HIT   {src}  p.{pg}  (rank #{rank})")
                else:
                    print(f"  ❌ SOURCE MISS  {src}  p.{pg}")

        if exp_answer:
            hit = answer_hit(retrieved, exp_answer)
            symbol = "✅" if hit else "❌"
            label  = "CONTENT HIT  — answer phrase found in retrieved chunks" if hit \
                     else "CONTENT MISS — answer phrase not found in retrieved chunks"
            print(f"  {symbol} {label}")

    print("=" * W)
    print()


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

def check_one(
    question: str,
    q_meta: dict | None,
    chunks: list,
    index,
    bm25,
    k: int,
) -> bool:
    """Retrieve and print results for one question. Returns True on content hit."""
    retrieved = retrieve(question, index, chunks, k=k, user=RETRIEVAL_USER, bm25=bm25)
    print_results(question, retrieved, q_meta)
    exp = (q_meta or {}).get("expected_answer", "")
    return answer_hit(retrieved, exp) if exp else True


def check_all(questions: list[dict], chunks: list, index, bm25, k: int) -> None:
    """Run retrieval for every question and print a hit/miss summary table."""
    content_hits = content_misses = 0
    source_hits  = source_misses  = 0
    na_count = 0
    missed_ids: list[str] = []

    for q in questions:
        qid      = q.get("id", q["question"][:25])
        question = q["question"]
        cat      = q.get("category", "")
        sub      = q.get("sub_type", "")
        exp_ans  = q.get("expected_answer", "")
        exp      = expected_pairs(q)

        is_na = (
            cat in ("out_of_scope", "ambiguous")
            or sub in ("out_of_scope", "ambiguous", "on_topic_unanswerable")
            or (not exp_ans and not exp)
        )

        retrieved = retrieve(question, index, chunks, k=k, user=RETRIEVAL_USER, bm25=bm25)

        if is_na:
            status = "N/A "
            na_count += 1
        else:
            if exp_ans and answer_hit(retrieved, exp_ans):
                content_hits += 1
                status = " HIT"
            else:
                content_misses += 1
                status = "MISS"
                missed_ids.append(str(qid))

            if exp:
                retrieved_set = {(r["source"], r["page"]) for r in retrieved}
                if all(p in retrieved_set for p in exp):
                    source_hits += 1
                else:
                    source_misses += 1

        diff      = q.get("difficulty", "")[:6]
        cat_label = (cat or sub)[:14]
        q_prev    = question[:44]
        print(f"  [{str(qid)[:5]}] {status}  {cat_label:<14}  {diff:<6}  {q_prev}")

    total_content = content_hits + content_misses
    total_source  = source_hits  + source_misses

    print()
    if total_content:
        pct = content_hits / total_content
        print(f"  Content hit rate  (primary):   {content_hits}/{total_content}  ({pct:.1%})")
    if total_source:
        pct = source_hits / total_source
        print(f"  Source  hit rate  (secondary): {source_hits}/{total_source}  ({pct:.1%})  "
              f"[annotated page in top-{k}]")
    if na_count:
        print(f"  N/A (out_of_scope / ambiguous): {na_count}")
    if missed_ids:
        print(f"\n  Content misses: {', '.join(missed_ids)}")
        print("  Re-run with --id <id> or --question \"...\" to inspect a specific miss.")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval debugger — top-k chunks and hit/miss vs a questions file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python -m scripts.retrieval_check --id q09\n"
            "  uv run python -m scripts.retrieval_check --question \"What is femicide?\"\n"
            "  uv run python -m scripts.retrieval_check --all\n"
            "  uv run python -m scripts.retrieval_check --all --questions data/questions_1.json\n"
            "  uv run python -m scripts.retrieval_check --id q09 --k 10"
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--id",       metavar="ID",   help="Question id, e.g. q09 (questions.json only)")
    group.add_argument("--question", "-q", metavar="TEXT", help="Free-form question string")
    group.add_argument("--all",      "-a", action="store_true", help="Check every question in the file")
    parser.add_argument("--k",        type=int, default=5,
                        help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--questions", default=str(DATA_DIR / "questions.json"), metavar="PATH",
                        help="Path to question file (default: data/questions.json)")
    args = parser.parse_args()

    index_path = DATA_DIR / "my_index.faiss"
    if not index_path.exists():
        print("ERROR: FAISS index not found — run chunk.py first")
        sys.exit(1)

    questions_path = Path(args.questions)
    if not questions_path.exists():
        print(f"ERROR: questions file not found at {questions_path}")
        sys.exit(1)

    questions = load_questions(questions_path)
    q_index   = build_index(questions)

    print(f"Loading artifacts from {DATA_DIR} …")
    chunks, index, bm25 = load_artifacts(DATA_DIR)

    if args.all:
        print(f"\nChecking {len(questions)} questions (k={args.k}) "
              f"from {questions_path.name} …\n")
        check_all(questions, chunks, index, bm25, k=args.k)

    elif args.id:
        q_meta = q_index.get(args.id)
        if q_meta is None:
            print(f"ERROR: id '{args.id}' not found in {questions_path}")
            sys.exit(1)
        check_one(q_meta["question"], q_meta, chunks, index, bm25, k=args.k)

    else:
        question = args.question.strip()
        q_meta   = q_index.get(question.lower())
        check_one(question, q_meta, chunks, index, bm25, k=args.k)


if __name__ == "__main__":
    main()
