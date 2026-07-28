from __future__ import annotations

"""
scripts/extract.py

Extracts text from a folder of PDFs and saves the output as corpus.json
and sample.json. Auto-generates EXTRACTION_REPORT.md. Reads input/output
directories from environment variables (via .env).

Usage:
    uv run python scripts/extract.py
"""

import json
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# Constants
NEAR_EMPTY_THRESHOLD = 50
SAMPLE_PAGE_COUNT = 10
MIN_REPEAT_THRESHOLD = 10
LOG_SEPARATOR = "─" * 52


def configure_logging() -> None:
    """Configure root logger to stdout with a readable format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def detect_repeated_lines(
    raw_pages: list[dict], min_repeat: int = MIN_REPEAT_THRESHOLD
) -> set[str]:
    """
    Detect lines that repeat across many pages (likely headers/footers).

    Args:
        raw_pages: List of raw page dicts with 'text' field.
        min_repeat: Minimum number of pages a line must appear on to be flagged.

    Returns:
        Set of strings considered repeated headers/footers.
    """
    line_counts: Counter = Counter()
    for entry in raw_pages:
        seen_in_page: set[str] = set()
        for line in entry["text"].splitlines():
            stripped = line.strip()
            if len(stripped) > 3 and stripped not in seen_in_page:
                line_counts[stripped] += 1
                seen_in_page.add(stripped)
    return {line for line, count in line_counts.items() if count >= min_repeat}


def clean_text(raw: str, repeated_lines: set[str] | None = None) -> str:
    """
    Clean raw PDF text by removing common artifacts.

    Cleaning steps:
    - Remove repeated headers/footers detected across pages
    - Rejoin words split by hyphenated line breaks
    - Merge lines broken mid-sentence
    - Collapse multiple blank lines into one
    - Collapse multiple spaces into one

    Args:
        raw: Raw text string extracted from a PDF page.
        repeated_lines: Set of header/footer strings to remove.

    Returns:
        Cleaned text string.
    """
    if repeated_lines:
        filtered_lines = [
            line for line in raw.splitlines()
            if line.strip() not in repeated_lines
        ]
        raw = "\n".join(filtered_lines)

    # Rejoin hyphenated line breaks (e.g. "impor-\ntant" -> "important")
    text = re.sub(r"-\n(\w)", r"\1", raw)
    # Merge lines broken mid-sentence (no punctuation at end)
    text = re.sub(r"(?<![.!?:])\n(?!\n)(?!\s*[-•*\d])", " ", text)
    # Collapse 3+ blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    # Strip each line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    return text.strip()


def _extract_tables_as_text(pdf_path: Path, page_num: int) -> str:
    """Extract tables from a PDF page using pdfplumber and return formatted text.

    Each table row is formatted as "Header1: value1, Header2: value2, ..." so
    the resulting text preserves the row→column relationship that PyMuPDF's
    linearised get_text() loses (e.g. Annex tables where country names sit in
    column 1 and statistics in columns 2–5).

    Returns an empty string when no tables are found or extraction fails.
    """
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            page = pdf.pages[page_num]
            tables = page.extract_tables()
            if not tables:
                return ""

            parts: list[str] = []
            for table in tables:
                if not table:
                    continue
                # First non-empty row is treated as the header row
                headers = [str(cell or "").strip() for cell in table[0]]
                for row in table[1:]:
                    cells = [str(cell or "").strip() for cell in row]
                    if not any(cells):
                        continue
                    row_text = ", ".join(
                        f"{h}: {v}"
                        for h, v in zip(headers, cells)
                        if h and v
                    )
                    if row_text:
                        parts.append(row_text)

            return "\n".join(parts)
    except Exception as exc:
        log.debug(
            "pdfplumber table extraction failed %s p%d: %s",
            pdf_path.name, page_num + 1, exc,
        )
        return ""


def extract_folder(input_dir: Path) -> list[dict]:
    """
    Extract text from all PDFs in a directory.
    First pass extracts raw text, second pass removes repeated headers/footers.

    Args:
        input_dir: Directory containing PDF files.

    Returns:
        Combined list of cleaned page dicts.
    """
    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        log.warning("No PDF files found in %s", input_dir)
        return []

    log.info("Found %d PDF file(s) in %s", len(pdf_files), input_dir)

    # First pass — extract raw text without cleaning
    raw_pages: list[dict] = []
    for pdf_path in pdf_files:
        log.info("Extracting: %s", pdf_path.name)
        try:
            doc = fitz.open(str(pdf_path))
            try:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text()

                    # Append structured table text so column→row associations
                    # (e.g. "Member State: Sweden, Median duration: 00:12:21")
                    # are preserved alongside the raw PyMuPDF output.
                    table_text = _extract_tables_as_text(pdf_path, page_num)
                    if table_text:
                        text = text + "\n\n" + table_text

                    raw_pages.append(
                        {
                            "source": pdf_path.name,
                            "page": page_num + 1,
                            "text": text,
                        }
                    )
            finally:
                doc.close()
        except Exception as exc:
            log.error("Failed to process %s: %s", pdf_path.name, exc)
        log.info("  → %d pages extracted so far", len(raw_pages))

    # Detect repeated headers/footers across all pages
    log.info("Detecting repeated headers/footers...")
    repeated_lines = detect_repeated_lines(raw_pages)
    log.info("  → %d repeated lines flagged for removal", len(repeated_lines))

    # Second pass — clean each page using detected repeated lines
    all_pages: list[dict] = []
    for entry in raw_pages:
        cleaned = clean_text(entry["text"], repeated_lines)
        all_pages.append(
            {
                "source": entry["source"],
                "page": entry["page"],
                "char_count": len(cleaned),
                "text": cleaned,
            }
        )

    return all_pages


def compute_stats(corpus: list[dict]) -> dict:
    """
    Compute corpus statistics.

    Args:
        corpus: List of page dicts.

    Returns:
        Dict with stats and near_empty_pages list.
    """
    sources = {entry["source"] for entry in corpus}
    total_pages = len(corpus)
    total_chars = sum(entry["char_count"] for entry in corpus)
    near_empty = [e for e in corpus if e["char_count"] < NEAR_EMPTY_THRESHOLD]
    avg_chars = total_chars / total_pages if total_pages else 0.0

    return {
        "num_documents": len(sources),
        "total_pages": total_pages,
        "total_chars": total_chars,
        "avg_chars_per_page": round(avg_chars, 1),
        "near_empty_count": len(near_empty),
        "near_empty_pages": near_empty,
    }


def print_stats(stats: dict) -> None:
    """
    Print corpus statistics using structured log format.

    Args:
        stats: Dict returned by compute_stats().
    """
    log.info(LOG_SEPARATOR)
    log.info("  %-30s %d", "Documents (PDFs):", stats["num_documents"])
    log.info("  %-30s %d", "Total pages:", stats["total_pages"])
    log.info("  %-30s %d", "Total characters:", stats["total_chars"])
    log.info("  %-30s %.1f", "Avg characters per page:", stats["avg_chars_per_page"])
    log.info("  %-30s %d", "Near-empty pages (< 50 chars):", stats["near_empty_count"])
    log.info(LOG_SEPARATOR)

    if stats["near_empty_pages"]:
        log.info("Near-empty pages:")
        for entry in stats["near_empty_pages"]:
            log.info(
                "  %s  page %d  (%d chars)",
                entry["source"],
                entry["page"],
                entry["char_count"],
            )


def generate_report(stats: dict, output_dir: Path) -> None:
    """
    Auto-generate EXTRACTION_REPORT.md from computed stats.

    Args:
        stats: Dict returned by compute_stats().
        output_dir: Output directory (used to locate scripts/ folder).
    """
    near_empty_lines = "\n".join(
        f"- `{e['source']}` — page {e['page']} ({e['char_count']} chars)"
        for e in stats["near_empty_pages"]
    ) or "_None found._"

    report = f"""# Extraction Report

## Corpus Statistics

| Metric | Value |
|---|---|
| Documents (PDFs) | {stats["num_documents"]} |
| Total pages | {stats["total_pages"]} |
| Total characters | {stats["total_chars"]} |
| Avg characters per page | {stats["avg_chars_per_page"]} |
| Near-empty pages (< 50 chars) | {stats["near_empty_count"]} |

## Near-Empty Pages Flagged

The following pages had fewer than 50 characters after cleaning:

{near_empty_lines}

## Text Cleaning Applied

- Repeated headers/footers removed (lines appearing on {MIN_REPEAT_THRESHOLD}+ pages)
- Hyphenated line breaks rejoined (e.g. "impor-\\ntant" → "important")
- Mid-sentence line breaks merged into single space
- Multiple blank lines collapsed into one
- Multiple spaces collapsed into one

## Configuration

- Library used: PyMuPDF (fitz)
- Input directory: `{os.environ.get("PDF_INPUT_DIR", "data/pdfs")}`
- Output directory: `{os.environ.get("PDF_OUTPUT_DIR", "data")}`
- Near-empty threshold: {NEAR_EMPTY_THRESHOLD} characters
- Header/footer repeat threshold: {MIN_REPEAT_THRESHOLD} pages
"""

    report_path = output_dir.parent / "scripts" / "EXTRACTION_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    log.info("Report saved → %s", report_path)


def save_json(data: list[dict], path: Path) -> None:
    """
    Serialise data to a JSON file with UTF-8 encoding.

    Args:
        data: List of dicts to serialise.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    log.info("Saved %d entries → %s", len(data), path)


def main() -> None:
    """Entry point: extract PDFs, save outputs, generate report, print stats."""
    configure_logging()

    input_dir = Path(os.environ.get("PDF_INPUT_DIR", "data/pdfs"))
    output_dir = Path(os.environ.get("PDF_OUTPUT_DIR", "data"))

    if not input_dir.is_dir():
        log.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    corpus = extract_folder(input_dir)

    if not corpus:
        log.error("No pages extracted — aborting.")
        sys.exit(1)

    save_json(corpus, output_dir / "corpus.json")
    save_json(corpus[:SAMPLE_PAGE_COUNT], output_dir / "sample.json")

    stats = compute_stats(corpus)
    print_stats(stats)
    generate_report(stats, output_dir)


if __name__ == "__main__":
    main()
