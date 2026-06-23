"""
Healthcare Knowledge Navigator — XML Parser.

Wraps the Canonicalizer to process JATS XML files, providing validation
and quarantine logic for malformed documents.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from preprocessing.canonicalizer import CanonicalDocument, Canonicalizer

logger = logging.getLogger(__name__)


class XMLParser:
    """
    Parses JATS XML into canonical JSON, validating the result.
    """

    def __init__(self, invalid_json_dir: Path) -> None:
        self.canonicalizer = Canonicalizer()
        self.invalid_json_dir = invalid_json_dir

    def parse_file(self, xml_path: Path, output_path: Path) -> bool:
        """
        Parse an XML file and save the canonical JSON.

        Args:
            xml_path: Path to the input XML.
            output_path: Path where the output JSON should be saved.

        Returns:
            True if parsing was successful and valid, False otherwise.
        """
        pmcid = xml_path.stem

        try:
            xml_content = xml_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                xml_content = xml_path.read_text(encoding="latin-1")
            except Exception as e:
                logger.error("Failed to read %s: %s", pmcid, e)
                return False

        try:
            # Assembly
            doc = self.canonicalizer.process_document(pmcid, xml_content)
        except Exception as e:
            logger.error("Error processing %s: %s", pmcid, e)
            return False

        # Validation
        is_valid, errors = self._validate_document(doc)
        if not is_valid:
            logger.warning("Validation failed for %s: %s", pmcid, " | ".join(errors))
            # Save to quarantine directory
            self.invalid_json_dir.mkdir(parents=True, exist_ok=True)
            quarantine_path = self.invalid_json_dir / f"{pmcid}.json"
            self.canonicalizer.save_canonical_json(doc, quarantine_path)
            return False

        # Save to success directory
        try:
            self.canonicalizer.save_canonical_json(doc, output_path)
            logger.debug("Successfully parsed %s", pmcid)
            return True
        except Exception as e:
            logger.error("Failed to save JSON for %s: %s", pmcid, e)
            return False

    def _validate_document(self, doc: CanonicalDocument) -> tuple[bool, list[str]]:
        """
        Validate the canonical document meets minimum structural requirements.

        Checks:
        - metadata exists (at least pmcid and title)
        - title exists
        - abstract exists (warning only usually, but we check if it's there)
        - sections extracted
        """
        errors = []
        is_valid = True

        pmcid = doc.metadata.get("pmcid")
        if not pmcid:
            errors.append("Missing PMCID in metadata")
            is_valid = False

        title = doc.metadata.get("title")
        if not title:
            errors.append("Missing article title")
            is_valid = False

        if not doc.sections and not doc.abstract:
            errors.append("Document has no sections and no abstract")
            is_valid = False

        return is_valid, errors
