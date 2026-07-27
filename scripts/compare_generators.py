from __future__ import annotations

"""
scripts/compare_generators.py

Compare generation models on identical retrieved context.

The design decision under test is the *generator*, so retrieval is held fixed:
each question is retrieved once, and every model answers from byte-identical
passages. Without that control the comparison would confound generation quality
with retrieval variance, and a model would be rewarded for being run on a lucky
retrieval.

Judged three ways, because no single one is sufficient:

  correctness   an LLM judge scores the answer against the reference
  groundedness  does the answer stay inside the passages it was given?
                measured lexically — the fraction of the answer's content words
                that appear in the retrieved context. A cheap, model-free
                proxy for hallucination that cannot itself hallucinate.
  citation      does the answer name a source, as the system prompt requires?
  latency       seconds per answer, which is a real constraint locally

The judge is held constant across conditions (one model judging all), so judge
bias applies equally to every candidate and cancels in the comparison.

All models are local and free.

Usage:
    uv run python -m scripts.compare_generators
    uv run python -m scripts.compare_generators --models llama3.1:8b qwen3.5:4b
    uv run python -m scripts.compare_generators --limit 8
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import llm as _llm  # noqa: E402
from scripts.chunk import load_artifacts, retrieve  # noqa: E402

DATA_DIR = Path(os.environ.get("PDF_OUTPUT_DIR", "data"))

#: Candidates that actually emit an answer. qwen3.5:2b and qwen3.5:4b are
#: excluded deliberately: both are reasoning models that consume an entire
#: 800-token budget on internal reasoning and return an empty string, so they
#: cannot be compared on answer quality at any practical budget. That is a
#: finding, not a configuration error — see the report.
DEFAULT_MODELS = ["llama3.1:8b", "gemma4:e4b", "qwen3.5:9b"]

_ANSWER_SYSTEM = """\
You answer questions about EU women's safety law using ONLY the passages given.

Rules:
- Use only the numbered passages. Do not add outside knowledge.
- Cite the source document and page for every claim you make.
- If the passages do not contain the answer, say so plainly instead of guessing.
- Be concise: three sentences at most.
"""

_JUDGE_SYSTEM = """\
You grade a generated answer against a reference answer.

Reply with exactly one word:
  PASS — the generated answer conveys the reference answer's key facts
  FAIL — it misses, contradicts, or fabricates them
"""

_STOP = None


def stopset():
    global _STOP
    if _STOP is None:
        from nltk.corpus import stopwords
        _STOP = set(stopwords.words("english"))
    return _STOP


def groundedness(answer: str, context: str) -> float:
    """Fraction of the answer's content words that appear in the context.

    Model-free by design: a hallucination detector that is itself a language
    model can hallucinate. This cannot — it is a blunt instrument, but a
    trustworthy one, and it is only ever compared across conditions.
    """
    stop = stopset()
    words = {w for w in re.findall(r"[a-z]{4,}", answer.lower()) if w not in stop}
    if not words:
        return 0.0
    ctx = context.lower()
    return sum(1 for w in words if w in ctx) / len(words)


def has_citation(answer: str) -> bool:
    return bool(re.search(r"\.pdf|page\s+\d+|\[\d+\]|source", answer, re.IGNORECASE))


#: Reasoning models (the qwen3.5 family) spend their budget on internal
#: reasoning before emitting anything. At max_tokens=400 qwen3.5:4b consumed the
#: entire budget and returned an EMPTY string, which scored groundedness 0.00
#: and silently invalidated its whole column. A budget large enough for the
#: reasoning plus the answer is required for the comparison to be meaningful.
MAX_ANSWER_TOKENS = 1500


def ask(client, model: str, question: str, context: str) -> tuple[str, float, dict]:
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=MAX_ANSWER_TOKENS,
        messages=[
            {"role": "system", "content": _ANSWER_SYSTEM},
            {"role": "user",
             "content": f"PASSAGES:\n{context}\n\nQUESTION: {question}"},
        ],
    )
    dt = time.perf_counter() - t0
    usage = resp.usage
    return (resp.choices[0].message.content or "").strip(), dt, {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
    }


def judge(client, model: str, question: str, expected: str, got: str) -> str:
    # An empty or whitespace-only answer is a failure by definition. It must
    # never reach the judge: the local judge rates "   " as PASS, so asking it
    # would silently award credit for producing nothing.
    if not got or not got.strip():
        return "FAIL"

    resp = client.chat.completions.create(
        model=model, temperature=0, max_tokens=8,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content":
             f"QUESTION: {question}\n\nREFERENCE: {expected}\n\nGENERATED: {got}"},
        ],
    )
    return "PASS" if "PASS" in (resp.choices[0].message.content or "").upper() else "FAIL"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare local generation models")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--questions", default=str(DATA_DIR / "questions.json"))
    ap.add_argument("--output", default=str(DATA_DIR / "generator_comparison.json"))
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    ok, detail = _llm.health_check()
    print(f"Backend: {detail}")
    if not ok:
        sys.exit("Backend unavailable.")

    import requests
    tags = [m["name"] for m in
            requests.get(f"{_llm.OLLAMA_HOST}/api/tags", timeout=10).json()["models"]]
    models = args.models or [m for m in DEFAULT_MODELS if m in tags]
    missing = [m for m in models if m not in tags]
    if missing:
        print(f"Not installed, skipping: {', '.join(missing)}")
        models = [m for m in models if m in tags]
    if not models:
        sys.exit("No candidate models installed. Try: ollama pull llama3.1:8b")

    judge_model = args.judge_model or models[0]
    print(f"Candidates : {', '.join(models)}")
    print(f"Judge      : {judge_model} (held constant across conditions)\n")

    items = [it for it in json.loads(Path(args.questions).read_text())
             if it.get("expected_answer")]
    if args.limit:
        items = items[: args.limit]

    chunks, index, bm25 = load_artifacts(DATA_DIR)

    # Retrieve ONCE per question — every model sees identical context.
    print("Retrieving shared context …")
    shared: list[dict] = []
    for n, it in enumerate(items, 1):
        results = retrieve(it["question"], index, chunks, k=args.k, bm25=bm25,
                           expand=False)
        ctx = "\n\n".join(
            f"[{i}] Source: {r['source']} — Page {r['page']}\n{r['text']}"
            for i, r in enumerate(results, 1)
        )
        shared.append({"item": it, "context": ctx})
        print(f"  {n}/{len(items)}", end="\r", flush=True)
    print()

    client = _llm.get_client()
    rows = []

    for model in models:
        print(f"\n=== {model}")
        t_model = time.perf_counter()
        recs = []
        for n, s in enumerate(shared, 1):
            it, ctx = s["item"], s["context"]
            try:
                answer, dt, usage = ask(client, model, it["question"], ctx)
            except Exception as exc:
                print(f"  [{n}] generation failed: {exc}")
                continue
            verdict = judge(client, judge_model, it["question"],
                            it["expected_answer"], answer)
            if not answer.strip():
                print(f"  [{n}] EMPTY answer from {model} "
                      f"({usage['completion_tokens']} completion tokens spent)")

            recs.append({
                "id": it.get("id", f"q{n:02d}"),
                "empty": not answer.strip(),
                "sub_type": it.get("sub_type", ""),
                "verdict": verdict,
                "grounded": round(groundedness(answer, ctx), 4),
                "cited": has_citation(answer),
                "latency_s": round(dt, 2),
                "completion_tokens": usage["completion_tokens"],
                "answer": answer[:400],
            })
            print(f"  [{n}/{len(shared)}] {verdict}  grounded="
                  f"{recs[-1]['grounded']:.2f}  {dt:.1f}s", flush=True)

        if not recs:
            continue
        n = len(recs)
        rows.append({
            "model": model,
            "n": n,
            "pass_rate": round(sum(r["verdict"] == "PASS" for r in recs) / n, 4),
            "groundedness": round(sum(r["grounded"] for r in recs) / n, 4),
            "citation_rate": round(sum(r["cited"] for r in recs) / n, 4),
            "empty_answers": sum(r["empty"] for r in recs),
            "median_latency_s": round(
                sorted(r["latency_s"] for r in recs)[n // 2], 2),
            "mean_completion_tokens": round(
                sum(r["completion_tokens"] for r in recs) / n, 1),
            "total_s": round(time.perf_counter() - t_model, 1),
            "per_question": recs,
        })

    print("\n" + "=" * 78)
    print("GENERATION MODEL COMPARISON — identical retrieved context")
    print("=" * 78)
    print(f"  {'model':<16}{'pass':>8}{'grounded':>11}{'cited':>8}"
          f"{'median s':>10}{'out tok':>9}{'empty':>7}")
    print("  " + "-" * 81)
    for r in sorted(rows, key=lambda x: -x["pass_rate"]):
        print(f"  {r['model']:<16}{r['pass_rate']:>7.1%}{r['groundedness']:>11.3f}"
              f"{r['citation_rate']:>7.1%}{r['median_latency_s']:>10.1f}"
              f"{r['mean_completion_tokens']:>9.0f}{r['empty_answers']:>7}")
    print("=" * 78)
    print(f"  n = {rows[0]['n'] if rows else 0} questions · judge = {judge_model}")
    print("  Retrieval held identical across all conditions.")

    Path(args.output).write_text(
        json.dumps({"judge": judge_model, "k": args.k, "results": rows},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
