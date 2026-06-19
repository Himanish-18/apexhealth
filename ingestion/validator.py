"""
Healthcare Knowledge Navigator — XML Validator.

Validates downloaded PMC JATS XML files for structural integrity
and required content. Invalid files are quarantined to a separate
directory so they do not contaminate the corpus.

Validation checks:
    1. File exists and is non-empty.
    2. Valid XML (parseable by lxml).
    3. PMCID is present in the XML.
    4. Article title is present.
    5. Abstract exists (warning only — not all articles have one).
    6. File is not corrupted (minimum size check).
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

logger = logging.getLogger(__name__)

# Minimum file size (bytes) to consider a file non-corrupted
_MIN_FILE_SIZE = 100


@dataclass
class ValidationResult:
    """Result of validating a single XML file.

    Attributes:
        pmcid: The PMC identifier of the validated file.
        is_valid: Whether the file passed all required checks.
        errors: List of error messages for failed checks.
        warnings: List of warning messages for non-critical issues.
    """

    pmcid: str = ""
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class XMLValidator:
    """Validates PMC JATS XML files for structural integrity.

    Runs a series of checks on each XML file and returns a detailed
    validation result. Files that fail validation can be quarantined
    to a separate directory.

    Example::

        validator = XMLValidator()
        result = validator.validate("PMC1234567", Path("data/raw_xml/PMC1234567.xml"))
        if not result.is_valid:
            validator.quarantine(
                Path("data/raw_xml/PMC1234567.xml"),
                Path("data/logs/invalid"),
            )
    """

    def validate(self, pmcid: str, xml_path: Path) -> ValidationResult:
        """Run all validation checks on a JATS XML file.

        Args:
            pmcid: The expected PMC identifier.
            xml_path: Path to the XML file to validate.

        Returns:
            ValidationResult indicating pass/fail with details.
        """
        result = ValidationResult(pmcid=pmcid)

        # Check 1: File exists
        if not xml_path.exists():
            result.is_valid = False
            result.errors.append(f"File does not exist: {xml_path}")
            return result

        # Check 2: File is non-empty and not corrupted
        file_size = xml_path.stat().st_size
        if file_size == 0:
            result.is_valid = False
            result.errors.append("File is empty (0 bytes)")
            return result

        if file_size < _MIN_FILE_SIZE:
            result.is_valid = False
            result.errors.append(
                f"File is suspiciously small ({file_size} bytes), "
                f"minimum expected: {_MIN_FILE_SIZE} bytes"
            )
            return result

        # Read the file content
        try:
            xml_content = xml_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Try with latin-1 as fallback
            try:
                xml_content = xml_path.read_text(encoding="latin-1")
            except Exception as e:
                result.is_valid = False
                result.errors.append(f"Cannot read file: {e}")
                return result

        # Check 3: Valid XML (parseable)
        try:
            # Use lxml's lenient recovery parser for JATS quirks
            parser = etree.XMLParser(recover=True, encoding="utf-8")
            tree = etree.fromstring(xml_content.encode("utf-8"), parser=parser)
        except etree.XMLSyntaxError as e:
            result.is_valid = False
            result.errors.append(f"Invalid XML syntax: {e}")
            return result

        if tree is None:
            result.is_valid = False
            result.errors.append("XML parsing returned None (severely malformed)")
            return result

        # Check 4: PMCID present in the XML
        pmc_id_found = self._check_pmcid(tree, pmcid)
        if not pmc_id_found:
            result.is_valid = False
            result.errors.append(
                f"PMCID '{pmcid}' not found in the XML article-id elements"
            )

        # Check 5: Title present
        title_found = self._check_title(tree)
        if not title_found:
            result.is_valid = False
            result.errors.append("Article title not found in the XML")

        # Check 6: Abstract (warning only)
        abstract_found = self._check_abstract(tree)
        if not abstract_found:
            result.warnings.append("No abstract found (non-critical)")

        if result.is_valid:
            logger.debug("Validation passed for %s", pmcid)
        else:
            logger.warning(
                "Validation FAILED for %s: %s", pmcid, "; ".join(result.errors)
            )

        return result

    def validate_content(self, pmcid: str, xml_content: str) -> ValidationResult:
        """Validate XML content directly (without reading from file).

        Args:
            pmcid: The expected PMC identifier.
            xml_content: Raw XML string to validate.

        Returns:
            ValidationResult indicating pass/fail with details.
        """
        result = ValidationResult(pmcid=pmcid)

        if not xml_content or not xml_content.strip():
            result.is_valid = False
            result.errors.append("XML content is empty")
            return result

        if len(xml_content) < _MIN_FILE_SIZE:
            result.is_valid = False
            result.errors.append(
                f"XML content is suspiciously small ({len(xml_content)} bytes)"
            )
            return result

        # Parse XML
        try:
            parser = etree.XMLParser(recover=True, encoding="utf-8")
            tree = etree.fromstring(xml_content.encode("utf-8"), parser=parser)
        except etree.XMLSyntaxError as e:
            result.is_valid = False
            result.errors.append(f"Invalid XML syntax: {e}")
            return result

        if tree is None:
            result.is_valid = False
            result.errors.append("XML parsing returned None")
            return result

        # PMCID check
        if not self._check_pmcid(tree, pmcid):
            result.is_valid = False
            result.errors.append(f"PMCID '{pmcid}' not found in the XML")

        # Title check
        if not self._check_title(tree):
            result.is_valid = False
            result.errors.append("Article title not found in the XML")

        # Abstract check (warning only)
        if not self._check_abstract(tree):
            result.warnings.append("No abstract found (non-critical)")

        return result

    @staticmethod
    def quarantine(xml_path: Path, invalid_dir: Path) -> Path | None:
        """Move an invalid XML file to the quarantine directory.

        Args:
            xml_path: Path to the invalid XML file.
            invalid_dir: Directory to move the file to.

        Returns:
            Path to the quarantined file, or None if the source doesn't exist.
        """
        if not xml_path.exists():
            logger.warning("Cannot quarantine — file does not exist: %s", xml_path)
            return None

        invalid_dir.mkdir(parents=True, exist_ok=True)
        dest = invalid_dir / xml_path.name

        shutil.move(str(xml_path), str(dest))
        logger.info("Quarantined invalid file: %s -> %s", xml_path, dest)
        return dest

    # ── Private Helpers ──────────────────────────────────────────────

    @staticmethod
    def _check_pmcid(tree: etree._Element, expected_pmcid: str) -> bool:
        """Check if the expected PMCID exists in article-id elements."""
        numeric_id = expected_pmcid.replace("PMC", "")

        # Search for <article-id pub-id-type="pmc">
        for article_id in tree.iter("article-id"):
            pub_id_type = article_id.get("pub-id-type", "")
            text = (article_id.text or "").strip()

            if pub_id_type == "pmc" and (
                text == numeric_id or text == expected_pmcid
            ):
                return True

        return False

    @staticmethod
    def _check_title(tree: etree._Element) -> bool:
        """Check if an article-title element exists with content."""
        for title in tree.iter("article-title"):
            if title.text and title.text.strip():
                return True
            # Check for mixed content (e.g., <italic> inside title)
            full_text = etree.tostring(title, method="text", encoding="unicode")
            if full_text and full_text.strip():
                return True
        return False

    @staticmethod
    def _check_abstract(tree: etree._Element) -> bool:
        """Check if an abstract element exists."""
        for abstract in tree.iter("abstract"):
            full_text = etree.tostring(abstract, method="text", encoding="unicode")
            if full_text and full_text.strip():
                return True
        return False
