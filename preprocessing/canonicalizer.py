"""
Healthcare Knowledge Navigator — Canonicalizer.

Defines the Canonical JSON schema and provides the orchestrator class
to assemble extracted metadata, tables, figures, and sections into
the final CanonicalDocument format.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag

from ingestion.metadata_extractor import MetadataExtractor
from preprocessing.parser_utils import extract_text_from_element
from preprocessing.section_extractor import SectionExtractor, SectionSchema
from preprocessing.table_extractor import TableExtractor, TableSchema


@dataclass
class FigureSchema:
    """Represents an extracted figure."""

    id: str
    caption: str


@dataclass
class CanonicalDocument:
    """
    The canonical JSON schema for a parsed medical document.
    This acts as the single source of truth for all downstream tasks.
    """

    metadata: dict[str, Any]
    abstract: str
    sections: list[SectionSchema]
    tables: list[TableSchema]
    figures: list[FigureSchema]
    references: list[str]


class Canonicalizer:
    """
    Assembles extracted components into the CanonicalDocument format.
    """

    def __init__(self) -> None:
        self.metadata_extractor = MetadataExtractor()
        self.table_extractor = TableExtractor()
        self.section_extractor = SectionExtractor()

    def process_document(self, pmcid: str, xml_content: str) -> CanonicalDocument:
        """
        Process the raw XML content and assemble the canonical document.

        Args:
            pmcid: The PMC identifier.
            xml_content: Raw JATS XML string.

        Returns:
            CanonicalDocument instance.
        """
        # Parse once
        soup = BeautifulSoup(xml_content, "lxml-xml")

        # 1. Metadata
        paper_metadata = self.metadata_extractor.extract(pmcid, xml_content)
        
        # 2. Abstract
        abstract = ""
        abstract_tag = soup.find("abstract")
        if abstract_tag:
            abstract = extract_text_from_element(abstract_tag)

        # 3. Tables
        tables = self.table_extractor.extract_tables(soup)

        # 4. Figures
        figures = self._extract_figures(soup)

        # 5. Sections
        body_tag = soup.find("body")
        sections = self.section_extractor.extract_sections(body_tag)

        # 6. References
        references = self._extract_references(soup)

        return CanonicalDocument(
            metadata=asdict(paper_metadata),
            abstract=abstract,
            sections=sections,
            tables=tables,
            figures=figures,
            references=references,
        )

    def save_canonical_json(self, doc: CanonicalDocument, output_path: Path) -> None:
        """
        Save the CanonicalDocument as JSON.

        Args:
            doc: CanonicalDocument to save.
            output_path: Path to the output JSON file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(doc), f, indent=2, ensure_ascii=False)

    def _extract_figures(self, soup: BeautifulSoup) -> list[FigureSchema]:
        """Extract figures from <fig> elements."""
        figures = []
        for fig in soup.find_all("fig"):
            fig_id = fig.get("id", f"fig_{len(figures)}")
            caption_tag = fig.find("caption")
            caption_text = extract_text_from_element(caption_tag) if caption_tag else ""
            figures.append(FigureSchema(id=fig_id, caption=caption_text))
        return figures

    def _extract_references(self, soup: BeautifulSoup) -> list[str]:
        """Extract references from <ref-list> elements."""
        references = []
        ref_list = soup.find("ref-list")
        if ref_list:
            for ref in ref_list.find_all("ref"):
                ref_text = extract_text_from_element(ref)
                if ref_text:
                    references.append(ref_text)
        return references
