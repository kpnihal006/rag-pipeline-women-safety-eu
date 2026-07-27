from __future__ import annotations

"""Tests for scripts/metadata.py — structured identifier extraction.

Dense embeddings lose exact identifiers: "14 June 2027" and "14 June 2032"
occupy nearly the same point in vector space. These tests pin the extraction
that lifts those identifiers into filterable fields.
"""

import pytest

from scripts.metadata import (
    extract_dates,
    extract_metadata,
    identifier_overlap,
    metadata_header,
    query_identifiers,
)

SAMPLE = (
    "Member States shall bring into force the laws necessary to comply with "
    "Directive (EU) 2024/1385 by 14 June 2027. Under Article 5, the Commission "
    "shall by 14 June 2032 carry out an evaluation. See also Regulation (EU) "
    "2022/2065 and Recital 17. Funding rose by 12.5% to EUR 4 500 000."
)


class TestDates:

    @pytest.mark.parametrize("text,iso", [
        ("due by 14 June 2027", "2027-06-14"),
        ("due by 1st March 2019", "2019-03-01"),
        ("on 2027-06-14 exactly", "2027-06-14"),
        ("dated 14/06/2027", "2027-06-14"),
    ])
    def test_formats_normalise_to_iso(self, text, iso):
        assert iso in extract_dates(text)[0]

    def test_surface_forms_are_kept(self):
        _, forms = extract_dates("deadline 14 June 2027")
        assert "14 June 2027" in forms

    def test_two_nearby_dates_are_both_captured(self):
        # The exact case dense retrieval conflates.
        iso, _ = extract_dates("by 14 June 2027 … and by 14 June 2032")
        assert "2027-06-14" in iso and "2032-06-14" in iso

    def test_invalid_numeric_date_is_rejected(self):
        assert extract_dates("ref 45/99/2027")[0] == []

    def test_no_dates_is_empty(self):
        assert extract_dates("no temporal content here") == ([], [])


class TestExtractMetadata:

    def test_all_identifier_classes(self):
        m = extract_metadata(SAMPLE)
        assert "2024/1385" in m["directives"]
        assert "2022/2065" in m["regulations"]
        assert "5" in m["articles"]
        assert "17" in m["recitals"]
        assert "12.5" in m["percentages"]
        assert m["money"]
        assert {2027, 2032} <= set(m["years"])

    def test_empty_fields_are_dropped(self):
        m = extract_metadata("Plain prose with no identifiers whatsoever.")
        assert "directives" not in m
        assert "articles" not in m

    def test_empty_text(self):
        assert extract_metadata("") == {}

    def test_table_flag(self):
        assert extract_metadata("a | b | c | d | e")["has_table"] is True
        assert "has_table" not in extract_metadata("prose without pipes")


class TestMetadataHeader:

    def test_header_puts_identifiers_in_context(self):
        # A bare "2027" is nearly signal-free to an embedding model; the same
        # token inside a labelled phrase is matchable.
        h = metadata_header(extract_metadata(SAMPLE))
        assert "Dates mentioned:" in h
        assert "14 June 2027" in h
        assert "Directive 2024/1385" in h
        assert "Article 5" in h

    def test_empty_metadata_gives_empty_header(self):
        assert metadata_header({}) == ""


class TestQueryMatching:

    def test_query_identifiers_are_extracted(self):
        q = query_identifiers("deadline for Directive (EU) 2024/1385?")
        assert "2024/1385" in q["directives"]

    def test_matching_chunk_scores_high(self):
        q = query_identifiers("transposition deadline for Directive (EU) 2024/1385?")
        assert identifier_overlap(q, extract_metadata(SAMPLE)) == 1.0

    def test_unrelated_chunk_scores_zero(self):
        q = query_identifiers("deadline for Directive (EU) 2024/1385?")
        other = extract_metadata("Article 9 of an unrelated text from 1999.")
        assert identifier_overlap(q, other) == 0.0

    def test_query_without_identifiers_does_not_boost(self):
        # Ordinary topical queries must be unaffected by the mechanism.
        q = query_identifiers("what protections exist for victims?")
        assert identifier_overlap(q, extract_metadata(SAMPLE)) == 0.0

    def test_year_only_partial_credit(self):
        q = query_identifiers("what happens in 2027?")
        score = identifier_overlap(q, extract_metadata(SAMPLE))
        assert 0.0 < score <= 1.0


class TestIdentifierBoostWiring:
    """The boost must actually be reachable from retrieval.

    An earlier version of this project documented an identifier boost that was
    never wired into `retrieve()` — the functions existed and were unit-tested,
    but nothing called them. These tests pin the wiring, not just the maths.
    """

    def test_retrieve_accepts_the_boost_parameter(self):
        import inspect
        from scripts.chunk import retrieve

        assert "identifier_boost" in inspect.signature(retrieve).parameters

    def test_retrieve_actually_calls_the_overlap_function(self):
        import inspect
        from scripts.chunk import retrieve

        src = inspect.getsource(retrieve)
        assert "identifier_overlap" in src
        assert "query_identifiers" in src

    def test_default_weight_is_on_the_cross_encoder_scale(self):
        from scripts.chunk import IDENTIFIER_BOOST

        # Cross-encoder logits span roughly [-11, 11]; a sub-1.0 weight is inert.
        assert IDENTIFIER_BOOST >= 3.0

    def test_bm25_indexes_the_metadata_header_when_present(self):
        import inspect
        from scripts.chunk import load_artifacts

        # Indexing only `text` leaves the metadata header invisible to BM25.
        assert "embed_text" in inspect.getsource(load_artifacts)
