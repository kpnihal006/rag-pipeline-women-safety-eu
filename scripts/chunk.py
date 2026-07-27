from __future__ import annotations

"""
scripts/chunk.py

RAG pipeline: corpus → sentence-aware chunks → embeddings → FAISS index
             → hybrid BM25+semantic retrieval → cross-encoder re-rank → answer

Steps (build):
  1. load_corpus   — reads corpus.json produced by extract.py
  2. chunk_corpus  — sentence-aware overlapping chunks, pdfplumber fallback
  3. embed_chunks  — local embedding model via app/llm.py (batched)
  4. build_index   — FAISS IndexFlatIP over L2-normalised vectors
  5. save_artifacts / load_artifacts — persist chunks.json + my_index.faiss

Steps (query):
  6. retrieve       — BM25 + FAISS fused via RRF, re-ranked by cross-encoder
  7. generate_answer — local chat model with retrieved context + history

Usage:
    uv run python scripts/chunk.py
"""

import json
import logging
import os
import sys
from pathlib import Path

import faiss
import nltk
import numpy as np
import openai
import pdfplumber
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from scripts.cost_function import track_cost
from scripts.metadata import identifier_overlap, query_identifiers
from app import llm as _llm

nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FALLBACK_CHAR_THRESHOLD = 100   # chars below which pdfplumber fallback triggers
CHUNK_SIZE    = 800             # target characters per chunk  (was 500)
CHUNK_OVERLAP = 200             # overlap carried forward       (was 100)

TOP_K             = 8           # final results returned to caller
RERANK_CANDIDATES = 150         # candidate pool before cross-encoder  (was 60)
HISTORY_TURNS     = 5           # max prior conversation turns sent to the LLM

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Model names resolve through app/llm.py, which defaults to the local Ollama
# backend. These module-level aliases are kept because other modules import
# them; call chat_model() / embed_model() directly in new code, since the
# backend can change per-process via LLM_BACKEND.
OPENAI_EMBED_MODEL  = _llm.embed_model()
OPENAI_MODEL        = _llm.chat_model()
EMBED_BATCH_SIZE    = 512       # well under the 2 048-input API limit

CONFIDENCE_THRESHOLD = 0.50     # top retrieval score below this → low-confidence flag

#: Weight added to a re-ranked score per unit of query/chunk identifier overlap.
#: Cross-encoder scores are logits roughly in [-11, 11], so the weight must be
#: on that scale to have any effect at all: a swept A/B over the benchmark shows
#: nothing changes below 3.0, and recall/MRR plateau from 5.0 through 20.0
#: (46.7% -> 53.3% recall@8, MRR 0.378 -> 0.444). 5.0 sits at the start of that
#: plateau. Set IDENTIFIER_BOOST=0 to disable.
#:
#: Caveat: tuned on the same 15-question benchmark it is evaluated on, so the
#: gain is an upper bound. The plateau is weak evidence it is not a knife-edge fit.
IDENTIFIER_BOOST = float(os.environ.get("IDENTIFIER_BOOST", "5.0"))

# ---------------------------------------------------------------------------
# Singletons — loaded once, reused across every request
# ---------------------------------------------------------------------------

_openai_client: openai.OpenAI | None = None
_cross_encoder: CrossEncoder | None = None

# Query-embedding cache: maps query string → L2-normalised float32 vector.
# Avoids re-embedding the same question across multiple requests / eval runs.
_embed_cache: dict[str, np.ndarray] = {}


def _get_openai_client() -> openai.OpenAI:
    """Client for the active backend (Ollama by default, OpenAI if selected)."""
    return _llm.get_client()


# ---------------------------------------------------------------------------
# Backend-scoped artifact names
# ---------------------------------------------------------------------------
# Ollama's nomic-embed-text produces 768-dim vectors; OpenAI's
# text-embedding-3-small produces 1536. Loading one index with the other
# backend's query embeddings fails at best and returns silent nonsense at
# worst, so each backend gets its own artifact files.

def _index_filename() -> str:
    return "my_index_ollama.faiss" if _llm.is_ollama() else "my_index.faiss"


def _chunks_filename() -> str:
    return "chunks_ollama.json" if _llm.is_ollama() else "chunks.json"


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        log.info("Loading cross-encoder: %s", CROSS_ENCODER_MODEL)
        try:
            _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        except Exception as exc:
            log.warning("Network load failed (%s) — retrying from local cache", exc)
            _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, local_files_only=True)
    return _cross_encoder


# ---------------------------------------------------------------------------
# Logging helper (used by main entry point only)
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


# ---------------------------------------------------------------------------
# Step 1 — Load corpus
# ---------------------------------------------------------------------------

def load_corpus(corpus_path: Path) -> list[dict]:
    """Load corpus.json produced by extract.py."""
    with open(corpus_path, encoding="utf-8") as fh:
        corpus = json.load(fh)
    log.info("Loaded %d pages from %s", len(corpus), corpus_path)
    return corpus


# ---------------------------------------------------------------------------
# Step 2 — Chunk with pdfplumber fallback
# ---------------------------------------------------------------------------

def _pdfplumber_page_text(pdf_path: Path, page_num: int) -> str:
    """Extract text from a single 0-indexed page using pdfplumber."""
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            return pdf.pages[page_num].extract_text() or ""
    except Exception as exc:
        log.debug("pdfplumber failed %s p%d: %s", pdf_path.name, page_num + 1, exc)
        return ""


def _resolve_text(entry: dict, input_dir: Path) -> tuple[str, str]:
    """Return (text, extractor) for a corpus page, using pdfplumber if PyMuPDF text is short."""
    stored = entry["text"]
    if len(stored.strip()) >= FALLBACK_CHAR_THRESHOLD:
        return stored, "pymupdf"

    pdf_path = input_dir / entry["source"]
    if not pdf_path.exists():
        log.warning("PDF not found for fallback: %s", pdf_path)
        return stored, "pymupdf"

    plumber_text = _pdfplumber_page_text(pdf_path, entry["page"] - 1)
    if len(plumber_text.strip()) > len(stored.strip()):
        return plumber_text, "pdfplumber"

    return stored, "pymupdf"


def _split(text: str) -> list[str]:
    """Sentence-aware overlapping chunker.

    Accumulates complete sentences up to CHUNK_SIZE chars, then carries
    forward tail sentences that fit within CHUNK_OVERLAP — no mid-sentence cuts.
    """
    if not text:
        return []

    sentences = sent_tokenize(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) > CHUNK_SIZE and current:
            chunks.append(" ".join(current))
            overlap, overlap_len = [], 0
            for s in reversed(current):
                if overlap_len + len(s) <= CHUNK_OVERLAP:
                    overlap.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            current, current_len = overlap, overlap_len

        current.append(sent)
        current_len += len(sent)

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_corpus(corpus: list[dict], input_dir: Path) -> list[dict]:
    """Chunk every page in the corpus. Returns a flat list of chunk dicts.

    In addition to per-page chunks, a cross-page stitch chunk is produced for
    every pair of consecutive pages from the same source.  This ensures answers
    that straddle a page boundary (e.g. a list whose last item continues on the
    next page) are always present in a single retrievable chunk.
    The stitch chunk carries the *first* page number and a distinct chunk_id so
    it can be identified in the retrieved results.
    """
    chunks: list[dict] = []
    fallback_count = 0

    # Resolve text for every entry first (needed for stitch look-ahead).
    resolved: list[tuple[str, str]] = []
    for entry in corpus:
        text, extractor = _resolve_text(entry, input_dir)
        if extractor == "pdfplumber":
            fallback_count += 1
        resolved.append((text, extractor))

    for idx, (entry, (text, extractor)) in enumerate(zip(corpus, resolved)):
        # Per-page chunks
        for i, chunk_text in enumerate(_split(text)):
            chunks.append({
                "chunk_id":    f"{entry['source']}::p{entry['page']}::c{i}",
                "source":      entry["source"],
                "page":        entry["page"],
                "chunk_index": i,
                "extractor":   extractor,
                "text":        chunk_text,
            })

        # Cross-page stitch: join tail of this page with head of next page
        # (same source, sequential page numbers only).
        if idx + 1 < len(corpus):
            next_entry = corpus[idx + 1]
            next_text, next_extractor = resolved[idx + 1]
            if (
                next_entry["source"] == entry["source"]
                and next_entry["page"] == entry["page"] + 1
                and text.strip()
                and next_text.strip()
            ):
                tail = text[-CHUNK_OVERLAP:].strip()
                head = next_text[:CHUNK_OVERLAP].strip()
                stitch_text = tail + " " + head
                chunks.append({
                    "chunk_id": (
                        f"{entry['source']}::p{entry['page']}-"
                        f"{next_entry['page']}::stitch"
                    ),
                    "source":      entry["source"],
                    "page":        entry["page"],
                    "chunk_index": -1,
                    "extractor":   extractor,
                    "text":        stitch_text,
                })

    log.info(
        "Chunked %d pages → %d chunks  (pdfplumber fallback on %d pages)",
        len(corpus), len(chunks), fallback_count,
    )
    return chunks


# ---------------------------------------------------------------------------
# Steps 3–4 — Embed + Index
# ---------------------------------------------------------------------------

def embed_chunks(chunks: list[dict], user: str | None = None) -> np.ndarray:
    """Embed all chunk texts in batches using OpenAI text-embedding-3-small."""
    client = _get_openai_client()
    texts = [c["text"] for c in chunks]
    log.info("Embedding %d chunks …", len(texts))

    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        response = client.embeddings.create(
            model=_llm.embed_model(),
            input=batch,
            encoding_format="float",
        )
        track_cost(response, call_type="embedding", user=user)
        vectors.extend([d.embedding for d in response.data])
        log.info("  embedded %d / %d", min(i + EMBED_BATCH_SIZE, len(texts)), len(texts))

    return np.array(vectors, dtype="float32")


def build_index(vectors: np.ndarray) -> faiss.Index:
    """Build a FAISS cosine-similarity index (L2-normalised inner product)."""
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    log.info("FAISS index built: %d vectors, dim=%d", index.ntotal, vectors.shape[1])
    return index


# ---------------------------------------------------------------------------
# Step 5 — Persist / load artifacts
# ---------------------------------------------------------------------------

def save_artifacts(chunks: list[dict], index: faiss.Index, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = output_dir / _chunks_filename()
    with open(chunks_path, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)
    log.info("Saved %d chunks → %s", len(chunks), chunks_path)

    index_path = output_dir / _index_filename()
    faiss.write_index(index, str(index_path))
    log.info("Saved FAISS index → %s", index_path)


def load_artifacts(output_dir: Path) -> tuple[list[dict], faiss.Index, BM25Okapi]:
    """Load chunks + FAISS index from disk; precompute BM25 and warm up cross-encoder."""
    chunks_path = output_dir / _chunks_filename()
    index_path  = output_dir / _index_filename()

    with open(chunks_path, encoding="utf-8") as fh:
        chunks = json.load(fh)
    index = faiss.read_index(str(index_path))
    log.info("Loaded %d chunks and FAISS index from %s", len(chunks), output_dir)

    log.info("Precomputing BM25 …")
    _stopwords = set(stopwords.words("english"))
    # Index `embed_text` when the chunk carries a metadata header, so the
    # identifiers lifted out by scripts/metadata.py are searchable lexically
    # as well as densely. Indexing only `text` left half the enrichment inert.
    tokenized = [
        [w for w in word_tokenize((c.get("embed_text") or c["text"]).lower())
         if w.isalpha() and w not in _stopwords]
        for c in chunks
    ]
    bm25 = BM25Okapi(tokenized)
    for chunk, tok in zip(chunks, tokenized):
        chunk["_tokens"] = tok
    log.info("BM25 ready")

    _get_cross_encoder()  # warm up — eliminates cold-start latency on first query
    return chunks, index, bm25


# ---------------------------------------------------------------------------
# Step 6 — Retrieve + Generate
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    index: faiss.Index,
    chunks: list[dict],
    k: int = TOP_K,
    user: str | None = None,
    bm25: BM25Okapi | None = None,
    rerank: bool = True,
    expand: bool = True,
    identifier_boost: float = IDENTIFIER_BOOST,
) -> list[dict]:
    """Hybrid retrieval: BM25 + FAISS fused with RRF, re-ranked by cross-encoder.

    Each returned chunk dict includes a "score" key (cross-encoder logit) and a
    "low_confidence" bool on the first result when the top score is below
    CONFIDENCE_THRESHOLD — callers can surface a warning to the user.

    Query embeddings are cached in-process to avoid re-embedding the same question.
    Short or informal queries are rewritten into formal legal language before retrieval.
    """
    # Query expansion — rewrite short/informal queries into precise legal language.
    # Triggers when query is under 80 chars OR contains casual markers.
    # Costs ~$0.0001 per call; skipped for already-formal long queries.
    # Pass expand=False to skip this step (e.g. for latency-sensitive callers).
    _INFORMAL = ("what's", "whats", "what is the deal", "tell me about",
                 "how do i", "can i", "do i", "is there a")
    query_lower = query.lower()
    needs_expansion = expand and (len(query) < 80 or any(m in query_lower for m in _INFORMAL))
    if needs_expansion:
        try:
            _exp_client = _get_openai_client()
            _exp_resp = _exp_client.chat.completions.create(
                model=_llm.chat_model(),
                temperature=0,
                max_tokens=80,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Rewrite the user's question as a precise, formal query "
                            "suitable for searching EU legal and policy documents. "
                            "Output only the rewritten query, nothing else."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
            )
            expanded = _exp_resp.choices[0].message.content.strip()
            if expanded:
                log.debug("Query expanded: %r → %r", query, expanded)
                query = expanded
        except Exception as exc:
            log.warning("Query expansion failed (%s) — using original query", exc)

    _stopwords = set(stopwords.words("english"))
    query_tokens = [
        w for w in word_tokenize(query.lower()) if w.isalpha() and w not in _stopwords
    ]

    # BM25 ranking
    if bm25 is None:
        bm25 = BM25Okapi([c.get("_tokens", c["text"].lower().split()) for c in chunks])
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_ranks  = np.argsort(bm25_scores)[::-1]

    # Semantic ranking via FAISS — use cache to avoid re-embedding repeated queries
    cache_key = f"{query}::{index.d}"
    if cache_key in _embed_cache:
        q_vec = _embed_cache[cache_key].copy()
        log.debug("Embedding cache hit for query: %s…", query[:60])
    else:
        client = _get_openai_client()
        embed_kwargs = {"model": _llm.embed_model(), "input": [query],
                        "encoding_format": "float"}
        if _llm.supports_dimensions():
            embed_kwargs["dimensions"] = index.d
        response = client.embeddings.create(**embed_kwargs)
        track_cost(response, call_type="embedding", user=user)
        q_vec = np.array([response.data[0].embedding], dtype="float32")
        faiss.normalize_L2(q_vec)
        _embed_cache[cache_key] = q_vec.copy()

    _, sem_indices = index.search(q_vec, len(chunks))
    sem_ranks = {int(idx): rank for rank, idx in enumerate(sem_indices[0]) if idx != -1}

    # Reciprocal Rank Fusion
    rrf_k = 60
    rrf: dict[int, float] = {}
    for rank, idx in enumerate(bm25_ranks):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    for idx, rank in sem_ranks.items():
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

    # Deduplicate by chunk text — prevents identical content from multiple PDFs
    # eating all top-k slots (e.g. same FRA sentence repeated across 4 files)
    seen_texts: set[str] = set()
    candidates: list[int] = []
    for idx in sorted(rrf, key=rrf.__getitem__, reverse=True):
        text = chunks[idx]["text"].strip()
        if text not in seen_texts:
            seen_texts.add(text)
            candidates.append(idx)
        if len(candidates) >= RERANK_CANDIDATES:
            break

    # Cross-encoder re-ranking (skipped when rerank=False for latency-sensitive callers)
    if rerank:
        pairs     = [(query, chunks[idx]["text"]) for idx in candidates]
        ce_scores = _get_cross_encoder().predict(pairs)
        reranked  = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)
    else:
        reranked = [(idx, rrf[idx]) for idx in candidates]

    # Identifier boost — applied to the re-ranked scores before the diversity
    # cap. When the query names a date, Directive, Regulation or Article, chunks
    # whose extracted metadata contains the same identifier are promoted.
    # `identifier_overlap` returns 0.0 when the query names no identifiers, so
    # ordinary topical queries are completely unaffected.
    if identifier_boost > 0.0:
        q_ids = query_identifiers(query)
        if any(q_ids.get(key) for key in
               ("dates", "years", "directives", "regulations", "articles")):
            scored = []
            for idx, score in reranked:
                overlap = identifier_overlap(q_ids, chunks[idx].get("metadata", {}))
                scored.append((idx, float(score) + identifier_boost * overlap))
            reranked = sorted(scored, key=lambda x: x[1], reverse=True)
            log.debug("Identifier boost applied for query ids: %s", q_ids)

    # Source diversity cap applied AFTER cross-encoding: cap each source at 3
    # slots in the final top-k so multi-hop questions always have 2+ sources,
    # while letting the cross-encoder see the full candidate pool first.
    MAX_PER_SOURCE = 3
    source_counts: dict[str, int] = {}
    diverse: list[tuple[int, float]] = []
    for idx, score in reranked:
        src = chunks[idx]["source"]
        if source_counts.get(src, 0) < MAX_PER_SOURCE:
            source_counts[src] = source_counts.get(src, 0) + 1
            diverse.append((idx, score))
        if len(diverse) >= k:
            break

    results = [{**chunks[idx], "score": float(score)} for idx, score in diverse]

    # Tag first result with low-confidence flag so callers can surface a warning
    if results and results[0]["score"] < CONFIDENCE_THRESHOLD:
        results[0]["low_confidence"] = True
        log.info(
            "Low-confidence retrieval: top score %.3f < threshold %.2f for query: %s…",
            results[0]["score"], CONFIDENCE_THRESHOLD, query[:60],
        )

    return results


_SYSTEM_PROMPT = (
    "You are a document assistant for women's safety laws and rights in the EU.\n"
    "Your ONLY source of facts is the excerpts provided below. "
    "Do NOT use knowledge from your training data.\n\n"
    "Rules:\n"
    "1. Every specific fact, statistic, date, name, or legal requirement you state "
    "MUST be present in at least one excerpt. When citing a key fact, quote the "
    "relevant passage directly (e.g. 'The excerpt states: \"...\"').\n"
    "2. You may synthesise information across multiple excerpts for comparison, "
    "yes/no, and multi-part questions — but only from what the excerpts contain. "
    "Do not fill gaps from your training data, even if you are confident.\n"
    "3. For yes/no questions: state Yes or No on the first line, then quote the "
    "supporting passage. If the excerpts imply but do not state the answer, say so "
    "explicitly before quoting.\n"
    "4. For comparative questions ('does X replace Y or work alongside it?'): "
    "reason step-by-step from what the excerpts say about each, then state your "
    "conclusion.\n"
    "5. If the question is completely unrelated to women's safety, gender equality, "
    "or EU law, respond with exactly: "
    "'This question is outside the scope of this corpus. "
    "I can only answer questions about women's safety and gender equality in the EU.'\n"
    "6. If the question is on-topic but the excerpts contain no relevant information, "
    "respond with exactly: "
    "'This specific information is not available in the corpus.'\n"
    "7. If the question is too vague to answer precisely, respond with exactly: "
    "'This question is too broad to answer precisely. Could you be more specific?'\n\n"
    "Excerpts:\n{context}"
)


def generate_answer(
    query: str,
    results: list[dict],
    user: str | None = None,
    history: list | None = None,
) -> str:
    """Generate an answer using retrieved context and optional conversation history.

    history is passed by Gradio as a list of {"role": ..., "content": ...} dicts
    (Gradio 4.x) or as a list of [user_msg, assistant_msg] pairs (older Gradio).
    """
    context = "\n\n---\n\n".join(
        f"[{r['source']} p.{r['page']}]\n{r['text']}" for r in results
    )
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT.format(context=context)}
    ]

    if history:
        for turn in history[-(HISTORY_TURNS * 2):]:
            if isinstance(turn, dict):
                messages.append({"role": turn["role"], "content": turn["content"]})
            else:
                user_msg, assistant_msg = turn
                messages.append({"role": "user",      "content": user_msg})
                messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({"role": "user", "content": query})

    client   = _get_openai_client()
    response = client.chat.completions.create(
        model=_llm.chat_model(),
        max_tokens=1024,
        temperature=0,
        messages=messages,
    )
    track_cost(response, call_type="chat", user=user)
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Entry point — build artifacts
# ---------------------------------------------------------------------------

def main() -> None:
    configure_logging()

    input_dir   = Path(os.environ["PDF_INPUT_DIR"])
    output_dir  = Path(os.environ["PDF_OUTPUT_DIR"])
    corpus_path = output_dir / "corpus.json"
    index_path  = output_dir / "my_index.faiss"
    chunks_path = output_dir / "chunks.json"

    if not corpus_path.exists():
        log.error("corpus.json not found at %s — run extract.py first", corpus_path)
        sys.exit(1)

    if index_path.exists() and chunks_path.exists():
        log.info("Artifacts found on disk — loading (skipping re-embed)")
        load_artifacts(output_dir)
        return

    corpus  = load_corpus(corpus_path)
    chunks  = chunk_corpus(corpus, input_dir)
    vectors = embed_chunks(chunks)
    index   = build_index(vectors)
    save_artifacts(chunks, index, output_dir)


if __name__ == "__main__":
    main()
