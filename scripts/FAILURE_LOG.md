# Retrieval & Generation Failure Log

Track every failure you find, what caused it, and what you changed to fix it.
Re-run the full eval after each fix to catch regressions.

**Workflow:**
1. `uv run python -m scripts.eval --no-judge` — get retrieval hit table
2. `uv run python -m scripts.retrieval_check --id <qid>` — inspect a miss
3. `uv run python -m scripts.eval` — re-run with LLM judge after a fix
4. Record below

---

## Failure entry template

```
### [qid] Short description of the question

**Category / Difficulty:** factual / medium
**Sub-type:** exact_term | multi_hop | temporal | …

**Failure type:** retrieval_miss | generation_error

**Symptom:**
What the pipeline actually returned vs what was expected.

**Root cause:**
Why it went wrong — chunk too small, BM25 missed an acronym, LLM ignored
context, wrong source indexed, answer spans two chunks, etc.

**Fix applied:**
What you changed (chunk size, overlap, prompt wording, TOP_K, etc.).

**Result:**
- Before: MISS / FAIL
- After:  HIT / PASS
- Regressions introduced: none | [list qids]

**Date:** YYYY-MM-DD
```

---

## Known systemic issues (from baseline RAGAS run)

| Metric | Score | Status |
|---|---|---|
| Faithfulness | 0.5083 | ❌ Poor — LLM adds facts not in retrieved chunks |
| Context precision | 0.8243 | ✅ Good — right chunks are being retrieved |
| Context recall | 0.6611 | ⚠️ Moderate — some relevant content is missed |

These are baselines before any optimisation. Run `ragas_eval.py` after each change to track improvement.

---

## Open failures

<!-- Add new entries here after running eval.py -->

---

## Resolved failures

<!-- Move resolved entries here once the fix is confirmed stable -->
