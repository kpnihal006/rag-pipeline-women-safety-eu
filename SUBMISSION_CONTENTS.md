# Submission contents

**Agentic RAG for EU women's safety policy** — Team DSA 8
Prerona Mitra · Nihal K P · Yogesh S J

Read [`reports/SCIENTIFIC_REPORT.md`](reports/SCIENTIFIC_REPORT.md) first, then
[`README.md`](README.md) to run it.

The system runs **entirely on free, locally-executed models** (Ollama). No paid
API key is required for the pipeline, the experiments, or the evaluation.

## Quick start

```bash
ollama serve &
ollama pull llama3.1:8b && ollama pull nomic-embed-text
uv sync
uv run python -c "from app import llm; print(llm.describe())"   # health check
uv run pytest tests/ -v                                          # 249 tests, offline
```

If this folder sits under an iCloud-synced directory, put the venv outside it:
`export UV_PROJECT_ENVIRONMENT=~/.venvs/rag-ws`

## Layout

| Path | What it is |
|---|---|
| `reports/SCIENTIFIC_REPORT.md` | Architecture, security, corpus processing, experiments, analysis |
| `reports/figures/` | 6 figures, regenerable via `scripts/make_plots.py` |
| `README.md` | Setup, run commands, experiment commands |
| `app/llm.py` | Backend resolution — Ollama by default |
| `app/mcp_server.py` | MCP server, 8 tools |
| `app/agents.py` | 5-agent team, role-scoped tool permissions |
| `app/main.py` | Supervisor router + human-in-the-loop gate |
| `app/security.py` | Injection, leakage, output validation, SSRF, traversal |
| `app/observability.py` | Span tracing, token/cost, prompt-window capture |
| `app/app.py`, `app/teams_bot.py` | Gradio UI, Teams webhook |
| `scripts/extract.py` | PDF extraction: PyMuPDF + pdfplumber fallback + table-aware |
| `scripts/chunk.py` | Sentence-aware chunking, embedding, hybrid retrieval |
| `scripts/metadata.py` | Date / directive / article identifier extraction |
| `scripts/reindex.py` | Re-embed without re-extracting (`--enrich`) |
| `scripts/diagnose_retrieval.py` | Stage-by-stage recall attribution |
| `scripts/experiment_grid.py` | Factorial: embedder × size × overlap × structure × k |
| `scripts/experiments.py` | Reranker / blend / candidate-pool sweep |
| `scripts/compare_generators.py` | Generation model comparison on identical context |
| `scripts/build_ground_truth.py` | Silver-standard eval set from the corpus |
| `scripts/eval.py`, `scripts/ragas_eval.py` | LLM-as-judge, RAGAS — both local |
| `scripts/make_plots.py` | Regenerates every report figure |
| `tests/` | 249 offline tests: no API key, no network, no index |
| `data/*.json` | Question set + all experiment results |
| `traces/*.json` | Real execution traces |

## Reproducing the experiments

```bash
uv run python -m scripts.diagnose_retrieval
uv run python -m scripts.experiment_grid --stage chunking
uv run python -m scripts.experiment_grid --stage embedding --output data/exp_embedding.json
uv run python -m scripts.experiment_grid --stage k         --output data/exp_k.json
uv run python -m scripts.experiment_grid --stage structure --output data/exp_structure.json
uv run python -m scripts.compare_generators
uv run python -m scripts.build_ground_truth --n 120
uv run python -m scripts.eval && uv run python scripts/ragas_eval.py
uv run python -m scripts.make_plots
```

## Not included

The FAISS index and source PDFs are omitted for size. Rebuild with
`uv run python -m scripts.extract && uv run python -m scripts.chunk`.
The test suite does not require them.
