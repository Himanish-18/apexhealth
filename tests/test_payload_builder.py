"""
Tests for payload builder (indexing.payload_builder).
Tests payload construction, point ID generation, and validation.
"""

import pytest

from indexing.payload_builder import (
    build_payload,
    generate_point_id,
    validate_payload,
)


class TestGeneratePointId:
    """Tests for deterministic point ID generation."""

    def test_deterministic_same_input(self):
        """Same chunk_id should always produce the same point ID."""
        id1 = generate_point_id("PMC123456_intro_001")
        id2 = generate_point_id("PMC123456_intro_001")
        assert id1 == id2

    def test_different_inputs_different_ids(self):
        """Different chunk_ids should produce different point IDs."""
        id1 = generate_point_id("PMC123456_intro_001")
        id2 = generate_point_id("PMC123456_intro_002")
        assert id1 != id2

    def test_returns_positive_integer(self):
        """Point IDs should be positive integers (63-bit)."""
        point_id = generate_point_id("PMC123456_intro_001")
        assert isinstance(point_id, int)
        assert point_id > 0

    def test_fits_in_63_bits(self):
        """Point IDs should fit in a signed 63-bit integer."""
        point_id = generate_point_id("PMC123456_intro_001")
        assert point_id <= 0x7FFFFFFFFFFFFFFF

    def test_empty_chunk_id_raises(self):
        """Empty chunk_id should raise ValueError."""
        with pytest.raises(ValueError, match="chunk_id must not be empty"):
            generate_point_id("")

    def test_unicode_chunk_id(self):
        """Unicode chunk IDs should work without error."""
        point_id = generate_point_id("PMC日本語_section_001")
        assert isinstance(point_id, int)
        assert point_id > 0


class TestBuildPayload:
    """Tests for payload construction."""

    def test_basic_payload_construction(self):
        """Should build a flat payload from chunk data."""
        chunk = {
            "chunk_id": "PMC123_intro_001",
            "content": "Some text content.",
            "metadata_header": "PMCID: PMC123",
            "metadata": {
                "pmcid": "PMC123",
                "title": "Test Paper",
                "journal": "Nature",
                "year": 2024,
                "section": "Introduction",
                "subsection": "",
                "section_path": ["Introduction"],
                "chunk_type": "TEXT",
                "token_count": 50,
            },
        }
        payload = build_payload(chunk)

        assert payload["chunk_id"] == "PMC123_intro_001"
        assert payload["pmcid"] == "PMC123"
        assert payload["title"] == "Test Paper"
        assert payload["journal"] == "Nature"
        assert payload["year"] == 2024
        assert payload["section"] == "Introduction"
        assert payload["chunk_type"] == "TEXT"
        assert payload["token_count"] == 50
        assert payload["content"] == "Some text content."
        assert payload["metadata_header"] == "PMCID: PMC123"
        assert payload["section_path"] == ["Introduction"]

    def test_document_metadata_fallback(self):
        """Should fall back to document_metadata when chunk metadata is sparse."""
        chunk = {
            "chunk_id": "PMC456_methods_001",
            "content": "Methods text.",
            "metadata": {},
        }
        doc_meta = {
            "pmcid": "PMC456",
            "title": "Fallback Paper",
            "journal": "Science",
            "year": 2023,
        }
        payload = build_payload(chunk, doc_meta)

        assert payload["pmcid"] == "PMC456"
        assert payload["title"] == "Fallback Paper"
        assert payload["journal"] == "Science"
        assert payload["year"] == 2023

    def test_missing_fields_default_to_empty(self):
        """Missing optional fields should default to empty strings/lists."""
        chunk = {"chunk_id": "PMC789_001", "metadata": {}}
        payload = build_payload(chunk)

        assert payload["article_type"] == ""
        assert payload["license"] == ""
        assert payload["subsection"] == ""
        assert payload["section_path"] == []


class TestValidatePayload:
    """Tests for payload validation."""

    def test_complete_payload_passes(self):
        """A payload with all required fields should have no missing fields."""
        payload = {
            "chunk_id": "PMC123_001",
            "pmcid": "PMC123",
            "title": "Test",
            "section": "Intro",
            "chunk_type": "TEXT",
            "token_count": 42,
        }
        missing = validate_payload(payload)
        assert missing == []

    def test_missing_fields_detected(self):
        """Missing required fields should be reported."""
        payload = {"chunk_id": "PMC123_001"}
        missing = validate_payload(payload)
        assert "pmcid" in missing
        assert "title" in missing

    def test_empty_values_detected(self):
        """Empty string values for required fields should be flagged."""
        payload = {
            "chunk_id": "PMC123_001",
            "pmcid": "",
            "title": "",
            "section": "Intro",
            "chunk_type": "TEXT",
            "token_count": 42,
        }
        missing = validate_payload(payload)
        assert "pmcid" in missing
        assert "title" in missing
