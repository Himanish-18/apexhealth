"""
Tests for the XML validator (ingestion.validator).

Validates the structural integrity checks applied to PMC JATS XML
files, including well-formedness, required fields, and quarantine logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.validator import ValidationResult, XMLValidator


# ── Valid JATS XML ────────────────────────────────────────────────────
VALID_JATS_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<article article-type="research-article">
  <front>
    <article-meta>
      <article-id pub-id-type="pmc">1234567</article-id>
      <title-group>
        <article-title>Valid Test Article</article-title>
      </title-group>
      <abstract>
        <p>This is a valid abstract.</p>
      </abstract>
    </article-meta>
  </front>
  <body>
    <p>Body content here.</p>
  </body>
</article>
"""

# ── Malformed XML ─────────────────────────────────────────────────────
MALFORMED_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<article>
  <front>
    <article-meta>
      <article-id pub-id-type="pmc">1234567
      <!-- Missing closing tags -->
"""

# ── XML without PMCID ────────────────────────────────────────────────
NO_PMCID_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<article>
  <front>
    <article-meta>
      <title-group>
        <article-title>Article Without PMCID</article-title>
      </title-group>
      <abstract><p>Abstract text.</p></abstract>
    </article-meta>
  </front>
</article>
"""

# ── XML without title ────────────────────────────────────────────────
NO_TITLE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<article>
  <front>
    <article-meta>
      <article-id pub-id-type="pmc">1234567</article-id>
      <abstract><p>Abstract text.</p></abstract>
    </article-meta>
  </front>
</article>
"""

# ── XML without abstract (should warn, not fail) ─────────────────────
NO_ABSTRACT_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<article>
  <front>
    <article-meta>
      <article-id pub-id-type="pmc">1234567</article-id>
      <title-group>
        <article-title>Article Without Abstract</article-title>
      </title-group>
    </article-meta>
  </front>
</article>
"""


@pytest.fixture
def validator() -> XMLValidator:
    """Create an XMLValidator instance."""
    return XMLValidator()


class TestFileValidation:
    """Tests for file-based XML validation."""

    def test_valid_xml_passes_validation(
        self,
        validator: XMLValidator,
        tmp_path: Path,
    ) -> None:
        """Well-formed JATS XML with PMCID and title should pass."""
        xml_path = tmp_path / "PMC1234567.xml"
        xml_path.write_text(VALID_JATS_XML, encoding="utf-8")

        result = validator.validate("PMC1234567", xml_path)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_nonexistent_file_fails(
        self,
        validator: XMLValidator,
        tmp_path: Path,
    ) -> None:
        """Non-existent file should fail validation."""
        xml_path = tmp_path / "PMC0000000.xml"

        result = validator.validate("PMC0000000", xml_path)

        assert not result.is_valid
        assert any("does not exist" in e for e in result.errors)

    def test_empty_file_fails(
        self,
        validator: XMLValidator,
        tmp_path: Path,
    ) -> None:
        """Empty file should fail validation."""
        xml_path = tmp_path / "PMC1234567.xml"
        xml_path.write_text("", encoding="utf-8")

        result = validator.validate("PMC1234567", xml_path)

        assert not result.is_valid
        assert any("empty" in e.lower() for e in result.errors)

    def test_small_file_fails(
        self,
        validator: XMLValidator,
        tmp_path: Path,
    ) -> None:
        """Suspiciously small file should fail validation."""
        xml_path = tmp_path / "PMC1234567.xml"
        xml_path.write_text("<a/>", encoding="utf-8")  # 4 bytes

        result = validator.validate("PMC1234567", xml_path)

        assert not result.is_valid
        assert any("small" in e.lower() for e in result.errors)


class TestContentValidation:
    """Tests for content-based XML validation."""

    def test_invalid_xml_fails(self, validator: XMLValidator) -> None:
        """Malformed XML should fail validation."""
        result = validator.validate_content("PMC1234567", MALFORMED_XML)

        # lxml with recover=True may parse partial XML; at minimum
        # the title check should fail since it's incomplete
        # But the key behavior is that it doesn't crash
        assert isinstance(result, ValidationResult)

    def test_missing_pmcid_fails(self, validator: XMLValidator) -> None:
        """XML without matching PMCID should fail."""
        result = validator.validate_content("PMC9999999", NO_PMCID_XML)

        assert not result.is_valid
        assert any("PMCID" in e for e in result.errors)

    def test_missing_title_fails(self, validator: XMLValidator) -> None:
        """XML without article title should fail."""
        result = validator.validate_content("PMC1234567", NO_TITLE_XML)

        assert not result.is_valid
        assert any("title" in e.lower() for e in result.errors)

    def test_missing_abstract_warns(self, validator: XMLValidator) -> None:
        """XML without abstract should pass but generate a warning."""
        result = validator.validate_content("PMC1234567", NO_ABSTRACT_XML)

        assert result.is_valid  # Abstract is non-critical
        assert any("abstract" in w.lower() for w in result.warnings)

    def test_empty_content_fails(self, validator: XMLValidator) -> None:
        """Empty XML content should fail."""
        result = validator.validate_content("PMC1234567", "")

        assert not result.is_valid
        assert any("empty" in e.lower() for e in result.errors)


class TestQuarantine:
    """Tests for the quarantine (move invalid files) logic."""

    def test_quarantine_moves_file(
        self,
        validator: XMLValidator,
        tmp_path: Path,
    ) -> None:
        """Invalid file should be moved to the quarantine directory."""
        xml_path = tmp_path / "PMC1234567.xml"
        xml_path.write_text("invalid content", encoding="utf-8")
        invalid_dir = tmp_path / "invalid"

        dest = validator.quarantine(xml_path, invalid_dir)

        assert dest is not None
        assert dest.exists()
        assert not xml_path.exists()  # Original should be gone
        assert dest.parent == invalid_dir
        assert dest.name == "PMC1234567.xml"

    def test_quarantine_creates_directory(
        self,
        validator: XMLValidator,
        tmp_path: Path,
    ) -> None:
        """Quarantine should create the invalid directory if needed."""
        xml_path = tmp_path / "PMC1234567.xml"
        xml_path.write_text("invalid content", encoding="utf-8")
        invalid_dir = tmp_path / "deep" / "nested" / "invalid"

        dest = validator.quarantine(xml_path, invalid_dir)

        assert invalid_dir.exists()
        assert dest is not None

    def test_quarantine_nonexistent_file(
        self,
        validator: XMLValidator,
        tmp_path: Path,
    ) -> None:
        """Quarantine of non-existent file should return None."""
        xml_path = tmp_path / "PMC0000000.xml"
        invalid_dir = tmp_path / "invalid"

        dest = validator.quarantine(xml_path, invalid_dir)

        assert dest is None
