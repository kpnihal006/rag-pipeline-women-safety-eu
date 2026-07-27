from __future__ import annotations

"""
scripts/build_ground_truth.py

Build a silver-standard evaluation set directly from the corpus, using the
local model only (no paid API).

WHY, AND WHAT IT IS NOT
-----------------------
The hand-written benchmark in `data/questions.json` has 17 scorable questions.
That is enough to report a headline number and far too few to *rank
configurations*: most category cells hold one question, so a single flip moves
a category score by 100 points. This set exists to give the ablations enough
statistical power to separate configurations that differ by a few points.

It is a **silver** standard, not gold, and it is deliberately kept separate
from the human set rather than merged, because it has a built-in bias: each
question is generated *from* a known chunk, so that chunk is by construction
answerable and unusually easy to retrieve. Absolute recall on this set reads
higher than real performance. What it measures honestly is the *relative*
ordering of retrieval configurations, which is what an ablation needs.

Two guards reduce (not eliminate) the circularity:

  1. **Paraphrase pressure.** The generator is told to avoid reusing the
     chunk's distinctive wording, so the question does not become a lexical
     copy that BM25 wins trivially.
  2. **Answerability filter.** A generated item is discarded unless the answer
     it claims is actually supported by the source chunk, checked by a second
     model pass. This removes hallucinated questions.

Sampling is stratified over documents and over chunks carrying structured
identifiers (dates, directive numbers, articles), so the set exercises the
identifier-heavy queries that the diagnosis flagged as weak.

Usage:
    uv run python -m scripts.build_ground_truth --n 120
    uv run python -m scripts.build_ground_truth --n 40 --no-verify
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import llm as _llm  # noqa: E402
from scripts.metadata import extract_metadata  # noqa: E402

DATA_DIR = Path(os.environ.get("PDF_OUTPUT_DIR", "data"))

#: Question styles, matched to the weaknesses the diagnosis surfaced.
STYLES = [
    ("factual", "a direct factual question answerable from the passage"),
    ("identifier",
     "a question about a specific date, deadline, directive number, or article "
     "number that appears in the passage"),
    ("paraphrased",
     "a question that asks about the passage's content WITHOUT reusing its "
     "distinctive vocabulary — describe the concept in different words"),
    ("negation",
     "a question phrased in the negative, e.g. asking what is NOT covered, "
     "excluded, or exempt according to the passage"),
    ("numeric",
     "a question about a figure, percentage, count, or monetary amount in the "
     "passage"),
]

_GEN_SYSTEM = """\
You write evaluation questions for a retrieval system over EU women's-safety \
law and policy documents.

Given a passage, write ONE question and its answer.

Rules:
- The question MUST be answerable using only the passage.
- The question must make sense on its own, to someone who has not seen the
  passage. Never write "according to the passage" or "in this document".
- Do NOT copy the passage's distinctive phrasing into the question. Someone
  searching a large corpus would use their own words.
- The answer must be one or two sentences, taken from the passage.
- If the passage is boilerplate (a table of contents, a header, a page number,
  a list of names, or otherwise carries no substantive content), reply exactly:
  SKIP

Reply as JSON only:
{"question": "...", "answer": "..."}
"""

_VERIFY_SYSTEM = """\
You check whether a passage really supports an answer.

Reply with exactly one word:
  SUPPORTED   — the passage states the answer, explicitly or near-verbatim
  UNSUPPORTED — the passage does not state it, or states something different
"""


def _chat(client, model: str, system: str, user: str, max_tokens: int = 400) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return (resp.choices[0].message.content or "").strip()


def _parse_json(raw: str) -> dict | None:
    if raw.upper().startswith("SKIP") or "SKIP" == raw.strip().upper():
        return None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    q, a = str(obj.get("question", "")).strip(), str(obj.get("answer", "")).strip()
    if len(q) < 15 or len(a) < 5:
        return None
    return {"question": q, "answer": a}


def substantive(chunk: dict) -> bool:
    """Filter out boilerplate before spending a model call on it."""
    text = chunk.get("text", "")
    if len(text) < 350:
        return False
    words = text.split()
    if len(words) < 60:
        return False
    # Mostly-numeric or mostly-punctuation blocks are tables of contents etc.
    alpha = sum(c.isalpha() for c in text)
    if alpha / max(1, len(text)) < 0.55:
        return False
    if text.lower().count("contents") > 2:
        return False
    return True


def stratified_sample(chunks: list[dict], n: int, seed: int = 20260727) -> list[int]:
    """Sample across documents, over-weighting identifier-bearing chunks."""
    rng = random.Random(seed)

    eligible = [i for i, c in enumerate(chunks) if substantive(c)]
    by_doc: dict[str, list[int]] = defaultdict(list)
    for i in eligible:
        by_doc[chunks[i]["source"]].append(i)

    with_ids: set[int] = set()
    for i in eligible:
        meta = extract_metadata(chunks[i]["text"])
        if meta.get("dates") or meta.get("directives") or meta.get("articles"):
            with_ids.add(i)

    picked: list[int] = []
    docs = sorted(by_doc)
    rng.shuffle(docs)

    # Round-robin across documents so no single large PDF dominates.
    pools = {d: rng.sample(by_doc[d], len(by_doc[d])) for d in docs}
    target_id_share = 0.5
    while len(picked) < n and any(pools.values()):
        for d in docs:
            if not pools[d]:
                continue
            want_id = (sum(1 for p in picked if p in with_ids)
                       < target_id_share * max(1, len(picked)))
            choice = None
            if want_id:
                for j, idx in enumerate(pools[d]):
                    if idx in with_ids:
                        choice = pools[d].pop(j)
                        break
            if choice is None:
                choice = pools[d].pop()
            picked.append(choice)
            if len(picked) >= n:
                break
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a silver-standard eval set")
    ap.add_argument("--n", type=int, default=120, help="target question count")
    ap.add_argument("--chunks", default=None, help="chunk store to sample from")
    ap.add_argument("--output", default=str(DATA_DIR / "ground_truth_silver.json"))
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the answerability check (faster, noisier)")
    ap.add_argument("--seed", type=int, default=20260727)
    args = ap.parse_args()

    ok, detail = _llm.health_check()
    print(f"Backend: {detail}")
    if not ok:
        sys.exit("Backend unavailable.")

    source = Path(args.chunks) if args.chunks else DATA_DIR / "chunks.json"
    chunks = json.loads(source.read_text(encoding="utf-8"))
    for c in chunks:
        c.pop("_tokens", None)
    print(f"Loaded {len(chunks):,} chunks from {source}")

    idxs = stratified_sample(chunks, args.n * 2, seed=args.seed)
    print(f"Sampled {len(idxs)} candidate chunks "
          f"across {len({chunks[i]['source'] for i in idxs})} documents\n")

    client = _llm.get_client()
    model = _llm.chat_model()
    rng = random.Random(args.seed)

    items: list[dict] = []
    skipped = unsupported = malformed = 0
    t0 = time.perf_counter()

    for n, idx in enumerate(idxs, 1):
        if len(items) >= args.n:
            break
        chunk = chunks[idx]
        style_name, style_desc = rng.choice(STYLES)

        prompt = (
            f"Write {style_desc}.\n\n"
            f"PASSAGE (from {chunk['source']}, page {chunk['page']}):\n"
            f"{chunk['text'][:2500]}"
        )
        try:
            raw = _chat(client, model, _GEN_SYSTEM, prompt)
        except Exception as exc:
            print(f"  [{n}] generation error: {exc}")
            continue

        parsed = _parse_json(raw)
        if parsed is None:
            if "SKIP" in raw.upper():
                skipped += 1
            else:
                malformed += 1
            continue

        if not args.no_verify:
            v = _chat(
                client, model, _VERIFY_SYSTEM,
                f"PASSAGE:\n{chunk['text'][:2500]}\n\n"
                f"ANSWER TO CHECK:\n{parsed['answer']}",
                max_tokens=10,
            )
            if "UNSUPPORTED" in v.upper():
                unsupported += 1
                continue

        meta = extract_metadata(chunk["text"])
        items.append({
            "id": f"s{len(items) + 1:03d}",
            "question": parsed["question"],
            "expected_answer": parsed["answer"],
            "sub_type": style_name,
            "difficulty": "silver",
            "gold_chunk_index": idx,
            "gold_chunk_id": chunk.get("chunk_id"),
            "expected_chunks": [{"source": chunk["source"], "page": chunk["page"]}],
            "source": chunk["source"],
            "page": chunk["page"],
            "has_identifiers": bool(
                meta.get("dates") or meta.get("directives") or meta.get("articles")
            ),
        })

        if len(items) % 10 == 0:
            rate = len(items) / max(time.perf_counter() - t0, 1e-6)
            print(f"  {len(items)}/{args.n} accepted "
                  f"({rate * 60:.0f}/min, {n} attempted)", flush=True)

    elapsed = time.perf_counter() - t0
    by_style: dict[str, int] = defaultdict(int)
    for it in items:
        by_style[it["sub_type"]] += 1

    print("\n" + "=" * 66)
    print("SILVER GROUND TRUTH")
    print("=" * 66)
    print(f"  accepted        {len(items)}")
    print(f"  attempted       {n}")
    print(f"  skipped (boilerplate)   {skipped}")
    print(f"  rejected (unsupported)  {unsupported}")
    print(f"  rejected (malformed)    {malformed}")
    print(f"  documents covered       {len({i['source'] for i in items})}")
    print(f"  with identifiers        {sum(1 for i in items if i['has_identifiers'])}")
    print(f"  elapsed                 {elapsed/60:.1f} min")
    print("\n  by style:")
    for s, c in sorted(by_style.items()):
        print(f"    {s:<14} {c}")
    print("=" * 66)

    out = Path(args.output)
    out.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved → {out}")
    print("\nNOTE: silver set — questions were generated from known chunks, so")
    print("absolute recall is optimistic. Use it to RANK configurations; report")
    print("the human set in data/questions.json as the headline number.")


if __name__ == "__main__":
    main()
