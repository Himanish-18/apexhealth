"""
Tests for the metadata extractor (ingestion.metadata_extractor).

Validates that all JATS XML metadata fields are correctly extracted
and persisted as JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingestion.metadata_extractor import MetadataExtractor, PaperMetadata


# ── Sample JATS XML with full metadata ───────────────────────────────
FULL_METADATA_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<article article-type="research-article"
         xmlns:xlink="http://www.w3.org/1999/xlink">
  <front>
    <journal-meta>
      <journal-title>Nature Medicine</journal-title>
    </journal-meta>
    <article-meta>
      <article-id pub-id-type="pmc">1234567</article-id>
      <article-id pub-id-type="doi">10.1038/s41591-024-00001-x</article-id>
      <title-group>
        <article-title>Advances in Cancer Immunotherapy: A Comprehensive Review</article-title>
      </title-group>
      <contrib-group>
        <contrib contrib-type="author">
          <name>
            <surname>Zhang</surname>
            <given-names>Wei</given-names>
          </name>
        </contrib>
        <contrib contrib-type="author">
          <name>
            <surname>Johnson</surname>
            <given-names>Emily R.</given-names>
          </name>
        </contrib>
        <contrib contrib-type="author">
          <name>
            <surname>Patel</surname>
            <given-names>Raj</given-names>
          </name>
        </contrib>
      </contrib-group>
      <pub-date pub-type="epub">
        <year>2024</year>
        <month>03</month>
        <day>15</day>
      </pub-date>
      <kwd-group>
        <kwd>immunotherapy</kwd>
        <kwd>cancer</kwd>
        <kwd>checkpoint inhibitors</kwd>
        <kwd>clinical trials</kwd>
      </kwd-group>
      <abstract>
        <p>This review covers recent advances in cancer immunotherapy.</p>
      </abstract>
      <permissions>
        <license xlink:href="https://creativecommons.org/licenses/by/4.0/">
          <license-p>This is an open access article.</license-p>
        </license>
      </permissions>
    </article-meta>
  </front>
</article>
"""

# ── Sample JATS XML with minimal/partial metadata ────────────────────
PARTIAL_METADATA_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<article>
  <front>
    <article-meta>
      <article-id pub-id-type="pmc">7654321</article-id>
      <title-group>
        <article-title>Minimal Metadata Paper</article-title>
      </title-group>
      <pub-date pub-type="ppub">
        <year>2023</year>
      </pub-date>
    </article-meta>
  </front>
</article>
"""


@pytest.fixture
def extractor() -> MetadataExtractor:
    """Create a MetadataExtractor instance."""
    return MetadataExtractor()


class TestExtractFullMetadata:
    """Tests for extracting complete metadata from JATS XML."""

    def test_extract_full_metadata(self, extractor: MetadataExtractor) -> None:
        """All fields should be extracted from well-formed JATS XML."""
        metadata = extractor.extract("PMC1234567", FULL_METADATA_XML)

        assert metadata.pmcid == "PMC1234567"
        assert metadata.doi == "10.1038/s41591-024-00001-x"
        assert "Advances in Cancer Immunotherapy" in metadata.title
        assert metadata.journal == "Nature Medicine"
        assert metadata.year == 2024
        assert metadata.article_type == "research-article"
        assert metadata.download_time  # Should be a non-empty ISO timestamp
        assert "creativecommons.org" in metadata.license

    def test_multiple_authors_extracted(self, extractor: MetadataExtractor) -> None:
        """All authors should be extracted in order."""
        metadata = extractor.extract("PMC1234567", FULL_METADATA_XML)

        assert len(metadata.authors) == 3
        assert metadata.authors[0] == "Wei Zhang"
        assert metadata.authors[1] == "Emily R. Johnson"
        assert metadata.authors[2] == "Raj Patel"

    def test_keywords_extracted(self, extractor: MetadataExtractor) -> None:
        """All keywords should be extracted."""
        metadata = extractor.extract("PMC1234567", FULL_METADATA_XML)

        assert len(metadata.keywords) == 4
        assert "immunotherapy" in metadata.keywords
        assert "cancer" in metadata.keywords
        assert "checkpoint inhibitors" in metadata.keywords
        assert "clinical trials" in metadata.keywords


class TestExtractPartialMetadata:
    """Tests for handling missing optional fields."""

    def test_extract_partial_metadata(self, extractor: MetadataExtractor) -> None:
        """Missing optional fields should default to empty values."""
        metadata = extractor.extract("PMC7654321", PARTIAL_METADATA_XML)

        assert metadata.pmcid == "PMC7654321"
        assert metadata.title == "Minimal Metadata Paper"
        assert metadata.year == 2023
        # Missing fields should be empty/default
        assert metadata.doi == ""
        assert metadata.authors == []
        assert metadata.journal == ""
        assert metadata.keywords == []
        assert metadata.license == ""

    def test_article_type_missing(self, extractor: MetadataExtractor) -> None:
        """Missing article-type attribute should return empty string."""
        metadata = extractor.extract("PMC7654321", PARTIAL_METADATA_XML)
        assert metadata.article_type == ""


class TestSaveMetadata:
    """Tests for persisting metadata as JSON."""

    def test_save_creates_json_file(
        self,
        extractor: MetadataExtractor,
        tmp_path: Path,
    ) -> None:
        """Saving metadata should create a JSON file with correct structure."""
        metadata = extractor.extract("PMC1234567", FULL_METADATA_XML)
        output_path = extractor.save(metadata, tmp_path)

        assert output_path.exists()
        assert output_path.name == "PMC1234567.json"

        # Load and verify JSON structure
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["pmcid"] == "PMC1234567"
        assert data["doi"] == "10.1038/s41591-024-00001-x"
        assert "Advances in Cancer Immunotherapy" in data["title"]
        assert len(data["authors"]) == 3
        assert data["year"] == 2024
        assert isinstance(data["keywords"], list)
        assert data["download_time"]  # Non-empty

    def test_save_creates_output_directory(
        self,
        extractor: MetadataExtractor,
        tmp_path: Path,
    ) -> None:
        """Save should create the output directory if it doesn't exist."""
        nested_dir = tmp_path / "deep" / "nested" / "metadata"
        metadata = extractor.extract("PMC1234567", FULL_METADATA_XML)

        output_path = extractor.save(metadata, nested_dir)

        assert output_path.exists()
        assert nested_dir.exists()
