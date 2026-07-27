from __future__ import annotations

"""
scripts/reindex.py

Re-embed an existing chunk store without re-extracting the PDFs.

Two things make this necessary:

  1. **Backend switch.** Ollama's nomic-embed-text produces 768-dim vectors
     against OpenAI's 1536, so moving to the local backend requires rebuilding
     the index. Re-running the full pipeline would also re-extract 2,247 pages,
     which is slow and unnecessary when the chunks have not changed.

  2. **Metadata enrichment.** `--enrich` prepends a natural-language header of
     the identifiers found in each chunk (dates, directive numbers, articles)
     before embedding, so an identifier sits in a context an embedding model
     can use rather than floating as a bare number. The header is embedded and
     kept in a separate field; the chunk's displayed `text` is untouched, so
     citations still quote the source faithfully.

Artifacts are written under backend-scoped filenames, so the OpenAI and Ollama
indexes coexist and cannot be loaded against the wrong query embeddings.

Usage:
    uv run python -m scripts.reindex                     # active backend
    uv run python -m scripts.reindex --enrich            # + metadata headers
    LLM_BACKEND=openai uv run python -m scripts.reindex --enrich
    uv run python -m scripts.reindex --source data/chunks.json --limit 500
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import llm as _llm  # noqa: E402
from scripts.chunk import (  # noqa: E402
    _chunks_filename,
    _index_filename,
    build_index,
)
from scripts.cost_function import track_cost  # noqa: E402
from scripts.metadata import extract_metadata, metadata_header  # noqa: E402

log = logging.getLogger(__name__)
DATA_DIR = Path(os.environ.get("PDF_OUTPUT_DIR", "data"))

#: Ollama's embedding endpoint handles one input at a time reliably; OpenAI
#: takes large batches. Keep them separate rather than tuning one compromise.
BATCH_OPENAI = 256
BATCH_OLLAMA = 32

#: nomic-embed-text has a 2048-token context (~8k chars). Stitched chunks and
#: table-derived text can exceed it, and prepending a metadata header makes
#: that more likely — so inputs are truncated before they are sent. Embedding
#: the first ~6k characters of an over-long chunk is strictly better than
#: aborting the run, which is what an unguarded request does.
MAX_EMBED_CHARS = 6000


def load_source_chunks(path: Path) -> list[dict]:
    chunks = json.loads(path.read_text(encoding="utf-8"))
    # `_tokens` is a BM25 cache written at load time; never persist it.
    for c in chunks:
        c.pop("_tokens", None)
    return chunks


def enrich(chunks: list[dict]) -> tuple[list[dict], int]:
    """Attach structured metadata and an embeddable header to every chunk."""
    enriched = 0
    for c in chunks:
        meta = extract_metadata(c["text"])
        if meta:
            c["metadata"] = meta
            header = metadata_header(meta)
            if header:
                c["embed_text"] = f"{header}\n\n{c['text']}"
                enriched += 1
    return chunks, enriched


def embed_all(chunks: list[dict], user: str | None = None) -> np.ndarray:
    """Embed every chunk with the active backend, in backend-sized batches."""
    client = _llm.get_client()
    model = _llm.embed_model()
    texts = [c.get("embed_text") or c["text"] for c in chunks]
    batch_size = BATCH_OLLAMA if _llm.is_ollama() else BATCH_OPENAI

    texts = [t[:MAX_EMBED_CHARS] for t in texts]
    n_truncated = sum(
        1 for c, t in zip(chunks, texts)
        if len(c.get("embed_text") or c["text"]) > MAX_EMBED_CHARS
    )
    if n_truncated:
        print(f"  {n_truncated} chunk(s) truncated to {MAX_EMBED_CHARS} chars "
              f"to fit the embedding context")

    vectors: list[list[float]] = []
    t0 = time.perf_counter()
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        try:
            resp = client.embeddings.create(
                model=model, input=batch, encoding_format="float"
            )
            vectors.extend([d.embedding for d in resp.data])
        except Exception as exc:
            # One bad input must not lose the whole run: fall back to
            # per-item embedding with harder truncation, and zero-fill only
            # the items that still refuse.
            print(f"\n  batch at {i} failed ({type(exc).__name__}) — "
                  f"retrying item by item", flush=True)
            dim = len(vectors[0]) if vectors else None
            for one in batch:
                try:
                    r1 = client.embeddings.create(
                        model=model, input=[one[:MAX_EMBED_CHARS // 2]],
                        encoding_format="float",
                    )
                    v = r1.data[0].embedding
                    dim = dim or len(v)
                    vectors.append(v)
                except Exception as exc2:
                    print(f"    skipped one chunk: {exc2}")
                    vectors.append([0.0] * (dim or 768))
            resp = None

        if resp is not None:
            try:
                track_cost(resp, call_type="embedding", user=user)
            except Exception:
                pass  # local backend reports no usage; never fail the reindex

        done = min(i + batch_size, len(texts))
        elapsed = time.perf_counter() - t0
        rate = done / max(elapsed, 1e-6)
        eta = (len(texts) - done) / max(rate, 1e-6)
        print(f"  embedded {done:,}/{len(texts):,}  "
              f"({rate:.0f}/s, eta {eta/60:.1f} min)", end="\r", flush=True)
    print()
    return np.array(vectors, dtype="float32")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    ap = argparse.ArgumentParser(description="Re-embed an existing chunk store")
    ap.add_argument("--source", default=None,
                    help="Chunk JSON to re-embed (default: data/chunks.json)")
    ap.add_argument("--enrich", action="store_true",
                    help="Prepend metadata headers before embedding")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N chunks (smoke test)")
    ap.add_argument("--suffix", default="",
                    help="Extra filename suffix, e.g. '_enriched'")
    args = ap.parse_args()

    backend = _llm.backend_name()
    ok, detail = _llm.health_check()
    print(f"Backend: {detail}")
    if not ok:
        print("Backend unavailable — aborting.")
        sys.exit(1)

    source = Path(args.source) if args.source else DATA_DIR / "chunks.json"
    if not source.exists():
        print(f"Source chunk store not found: {source}")
        sys.exit(1)

    chunks = load_source_chunks(source)
    if args.limit:
        chunks = chunks[: args.limit]
    print(f"Loaded {len(chunks):,} chunks from {source}")

    if args.enrich:
        chunks, n = enrich(chunks)
        print(f"Enriched {n:,} chunks with metadata headers "
              f"({n/len(chunks):.0%} carried identifiers)")

    vectors = embed_all(chunks)
    print(f"Embedded: {vectors.shape[0]:,} vectors, dim={vectors.shape[1]}")

    index = build_index(vectors)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stem_i = _index_filename().replace(".faiss", f"{args.suffix}.faiss")
    stem_c = _chunks_filename().replace(".json", f"{args.suffix}.json")
    index_path = DATA_DIR / stem_i
    chunks_path = DATA_DIR / stem_c

    faiss.write_index(index, str(index_path))
    chunks_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nWrote index  → {index_path}  ({index.ntotal:,} vectors, dim {index.d})")
    print(f"Wrote chunks → {chunks_path}")
    print(f"Backend: {backend} · embed model: {_llm.embed_model()}")


if __name__ == "__main__":
    main()
