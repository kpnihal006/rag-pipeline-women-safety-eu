from __future__ import annotations

"""
scripts/metadata.py

Structured metadata extraction for legal chunks.

Dense embeddings compress a passage into a direction in vector space, which is
excellent for topical similarity and poor for exact identifiers. "14 June 2027"
and "14 June 2032" embed almost identically, and a query asking for a
transposition deadline has no way to prefer one over the other. BM25 helps only
when the query happens to contain the literal token, which a paraphrased
question rarely does.

This module lifts those identifiers out of the prose into explicit fields:

    dates             ISO-normalised where possible, plus surface forms
    years             bare four-digit years mentioned
    directives        Directive (EU) 2024/1385 style references
    regulations       Regulation (EU) 2022/2065 style references
    articles          Article 5, Article 12a
    recitals          Recital 17
    percentages       12.5%
    money             EUR 4 500 000
    has_table         whether the chunk contains rendered table structure

Retrieval uses them two ways: as a lexical boost when a query mentions the same
identifier, and as extra indexed text so the identifier is embedded in a
context that makes its role explicit.

Usage:
    from scripts.metadata import extract_metadata, metadata_header
    meta = extract_metadata(chunk_text)
"""

import re
from typing import Any

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}
_MONTH_ALT = "|".join(_MONTHS)

# "14 June 2027" / "14th June 2027"
_RE_DMY = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_ALT})\s+((?:19|20)\d{{2}})\b",
    re.IGNORECASE,
)
# "June 14, 2027" / "June 2027"
_RE_MDY = re.compile(
    rf"\b({_MONTH_ALT})\s+(\d{{1,2}})?,?\s*((?:19|20)\d{{2}})\b",
    re.IGNORECASE,
)
# ISO "2027-06-14" and numeric "14/06/2027"
_RE_ISO = re.compile(r"\b((?:19|20)\d{2})-(\d{2})-(\d{2})\b")
_RE_NUMERIC = re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.]((?:19|20)\d{2})\b")

_RE_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")

# Directive (EU) 2024/1385 · Directive 2012/29/EU · Council Directive 2004/113/EC
_RE_DIRECTIVE = re.compile(
    r"\b(?:Council\s+|Commission\s+)?Directive\s*"
    r"(?:\(EU\)\s*|\(EC\)\s*)?(\d{4}/\d{1,4})(?:/(?:EU|EC|EEC))?",
    re.IGNORECASE,
)
_RE_REGULATION = re.compile(
    r"\b(?:Council\s+|Commission\s+)?Regulation\s*"
    r"(?:\(EU\)\s*|\(EC\)\s*)?(?:No\s*)?(\d{4}/\d{1,4}|\d{1,4}/\d{4})",
    re.IGNORECASE,
)
_RE_ARTICLE = re.compile(r"\bArticle\s+(\d{1,3}[a-z]?)\b", re.IGNORECASE)
_RE_RECITAL = re.compile(r"\bRecital\s+(\d{1,3})\b", re.IGNORECASE)
_RE_PERCENT = re.compile(r"\b(\d{1,3}(?:[.,]\d{1,2})?)\s?%")
_RE_MONEY = re.compile(
    r"(?:EUR|€|USD|\$)\s?\d[\d\s.,]*(?:\s?(?:million|billion|bn|m))?",
    re.IGNORECASE,
)

_TABLE_MARKER = "|"


def _norm_year(y: str) -> int:
    return int(y)


def extract_dates(text: str) -> tuple[list[str], list[str]]:
    """Return (iso_dates, surface_forms) found in the text."""
    iso: list[str] = []
    surface: list[str] = []

    for m in _RE_DMY.finditer(text):
        d, mon, y = m.group(1), m.group(2).lower(), m.group(3)
        iso.append(f"{_norm_year(y):04d}-{_MONTHS[mon]:02d}-{int(d):02d}")
        surface.append(m.group(0))

    for m in _RE_MDY.finditer(text):
        mon, d, y = m.group(1).lower(), m.group(2), m.group(3)
        if d:
            iso.append(f"{_norm_year(y):04d}-{_MONTHS[mon]:02d}-{int(d):02d}")
        else:
            iso.append(f"{_norm_year(y):04d}-{_MONTHS[mon]:02d}")
        surface.append(m.group(0))

    for m in _RE_ISO.finditer(text):
        iso.append(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
        surface.append(m.group(0))

    for m in _RE_NUMERIC.finditer(text):
        d, mon, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if 1 <= mon <= 12 and 1 <= d <= 31:
            iso.append(f"{_norm_year(y):04d}-{mon:02d}-{d:02d}")
            surface.append(m.group(0))

    return list(dict.fromkeys(iso)), list(dict.fromkeys(surface))


def _uniq(matches) -> list[str]:
    return list(dict.fromkeys(m.strip() for m in matches if str(m).strip()))


def extract_metadata(text: str) -> dict[str, Any]:
    """Extract structured identifiers from a chunk of legal text."""
    if not text:
        return {}

    iso_dates, date_forms = extract_dates(text)

    meta: dict[str, Any] = {
        "dates": iso_dates,
        "date_forms": date_forms,
        "years": sorted({int(y) for y in _RE_YEAR.findall(text)}),
        "directives": _uniq(_RE_DIRECTIVE.findall(text)),
        "regulations": _uniq(_RE_REGULATION.findall(text)),
        "articles": _uniq(a.lower() for a in _RE_ARTICLE.findall(text)),
        "recitals": _uniq(_RE_RECITAL.findall(text)),
        "percentages": _uniq(_RE_PERCENT.findall(text)),
        "money": _uniq(_RE_MONEY.findall(text)),
        "has_table": _TABLE_MARKER in text and text.count(_TABLE_MARKER) >= 4,
    }
    # Drop empties so the persisted chunks stay small.
    return {k: v for k, v in meta.items() if v}


def metadata_header(meta: dict[str, Any]) -> str:
    """A short natural-language prefix that makes identifiers embeddable.

    A bare "2027" carries almost no signal to an embedding model. The same
    token inside "Dates mentioned: 14 June 2027" sits in a context that a
    question about deadlines can actually match.
    """
    if not meta:
        return ""

    parts: list[str] = []
    if meta.get("date_forms"):
        parts.append("Dates mentioned: " + ", ".join(meta["date_forms"][:6]))
    if meta.get("directives"):
        parts.append("Directives cited: " + ", ".join(
            f"Directive {d}" for d in meta["directives"][:5]))
    if meta.get("regulations"):
        parts.append("Regulations cited: " + ", ".join(
            f"Regulation {r}" for r in meta["regulations"][:5]))
    if meta.get("articles"):
        parts.append("Articles referenced: " + ", ".join(
            f"Article {a}" for a in meta["articles"][:8]))
    if meta.get("percentages"):
        parts.append("Figures: " + ", ".join(f"{p}%" for p in meta["percentages"][:6]))
    if meta.get("has_table"):
        parts.append("Contains a data table.")

    return " | ".join(parts)


def query_identifiers(query: str) -> dict[str, list[str]]:
    """Identifiers present in a *query*, for lexical boosting at retrieval."""
    iso, forms = extract_dates(query)
    return {
        "dates": iso,
        "date_forms": forms,
        "years": [str(y) for y in sorted({int(y) for y in _RE_YEAR.findall(query)})],
        "directives": _uniq(_RE_DIRECTIVE.findall(query)),
        "regulations": _uniq(_RE_REGULATION.findall(query)),
        "articles": _uniq(a.lower() for a in _RE_ARTICLE.findall(query)),
    }


def identifier_overlap(query_ids: dict[str, list[str]], chunk_meta: dict) -> float:
    """Fraction of the query's identifiers that the chunk also contains.

    Returns 0.0 when the query names no identifiers, so ordinary topical
    queries are unaffected by the boost.
    """
    keys = ("dates", "years", "directives", "regulations", "articles")
    wanted = [(k, v) for k in keys for v in query_ids.get(k, [])]
    if not wanted:
        return 0.0

    hits = 0
    for key, value in wanted:
        have = chunk_meta.get(key, [])
        have_str = [str(x).lower() for x in have]
        if str(value).lower() in have_str:
            hits += 1
        elif key == "dates":
            # A query naming a year should match a chunk with a date in it.
            if any(str(value)[:4] == str(h)[:4] for h in have_str):
                hits += 0.5
    return hits / len(wanted)


__all__ = [
    "extract_metadata", "metadata_header", "extract_dates",
    "query_identifiers", "identifier_overlap",
]
