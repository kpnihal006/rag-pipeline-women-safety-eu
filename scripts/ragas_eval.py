from __future__ import annotations

"""
scripts/ragas_eval.py

Scores eval_results.json using RAGAS, driven by the **local** Ollama backend.

RAGAS instantiates its own judge LLM and embedding model and defaults to a
hosted provider, so both are wired explicitly here through LangChain adapters
pointed at Ollama's OpenAI-compatible endpoint. Nothing paid is called.

Input:  data/eval_results.json   — produced by eval.py
Output: data/ragas_scores.json

Score guide:  > 0.8 good | 0.5-0.8 moderate | < 0.5 poor

Usage:
    uv run python scripts/ragas_eval.py
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from ragas import EvaluationDataset, evaluate
from ragas.metrics import faithfulness, context_precision, context_recall

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import llm as _llm  # noqa: E402

load_dotenv()

DATA_DIR = Path(os.environ.get("PDF_OUTPUT_DIR", "data"))


def _local_judge():
    """RAGAS judge LLM + embeddings, both served locally by Ollama."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    base = f"{_llm.OLLAMA_HOST.rstrip('/')}/v1"
    chat = ChatOpenAI(model=_llm.chat_model(), base_url=base, api_key="ollama",
                      temperature=0, timeout=180)
    emb = OpenAIEmbeddings(model=_llm.embed_model(), base_url=base,
                           api_key="ollama", check_embedding_ctx_length=False)
    return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(emb)


def run_ragas(input_path: Path, output_path: Path) -> None:
    with open(input_path, encoding="utf-8") as fh:
        eval_data = json.load(fh)

    rows = eval_data["results"]
    limit = int(os.environ.get("RAGAS_LIMIT", "0"))
    if limit:
        rows = rows[:limit]
        print(f"  limited to the first {limit} questions (RAGAS_LIMIT)")

    dataset = EvaluationDataset.from_list([
        {
            "user_input":         r["question"],
            "response":           r["answer"],
            "retrieved_contexts": r["contexts"],
            "reference":          r.get("expected_answer", ""),
        }
        for r in rows
    ])

    print(f"\nRunning RAGAS on {len(rows)} questions …")

    judge_llm, judge_emb = _local_judge()
    print(f"  judge: {_llm.chat_model()}  ·  embeddings: {_llm.embed_model()} (local)")

    # A local backend serves one request at a time. RAGAS defaults to 16
    # concurrent workers with a 180s deadline, so every job queues behind the
    # others and times out — the first local run returned NaN for all three
    # metrics because all 60 jobs failed this way. Serialise and allow a
    # realistic per-call budget instead.
    from ragas.run_config import RunConfig

    run_config = RunConfig(
        timeout=int(os.environ.get("RAGAS_TIMEOUT", "900")),
        max_workers=int(os.environ.get("RAGAS_WORKERS", "1")),
        max_retries=2,
    )
    print(f"  run config: max_workers={run_config.max_workers}, "
          f"timeout={run_config.timeout}s (local backends are serial)")

    result = evaluate(
        dataset,
        metrics=[faithfulness, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_emb,
        run_config=run_config,
    )

    df = result.to_pandas()
    metric_cols = [c for c in df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")]

    print("\nRAGAS scores:")
    summary = {}
    for metric in metric_cols:
        raw = float(df[metric].mean())
        if raw != raw:  # NaN — every job for this metric failed
            summary[metric] = None
            print(f"  {metric:<22} FAILED — all jobs errored or timed out")
            continue
        score = round(raw, 4)
        summary[metric] = score
        bar    = "#" * max(0, int(score * 20))
        rating = "good" if score > 0.8 else "moderate" if score >= 0.5 else "poor"
        print(f"  {metric:<22} {score:.4f}  [{bar:<20}]  {rating}")

    if all(v is None for v in summary.values()):
        print("\n  Every metric failed. With a local backend this is almost"
              "\n  always concurrency: lower RAGAS_WORKERS or raise RAGAS_TIMEOUT.")

    output = {
        "summary":      summary,
        "per_question": df.to_dict(orient="records"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    print(f"\nScores saved → {output_path}")


if __name__ == "__main__":
    input_path  = DATA_DIR / "eval_results.json"
    output_path = DATA_DIR / "ragas_scores.json"

    if not input_path.exists():
        print(f"ERROR: {input_path} not found — run eval.py first")
        sys.exit(1)

    run_ragas(input_path, output_path)
