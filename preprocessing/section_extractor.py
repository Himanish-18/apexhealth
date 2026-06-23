"""
Healthcare Knowledge Navigator — Section Extractor.

Extracts sections and nested subsections from JATS XML documents,
preserving the structural hierarchy. Extracts the content of the
section while skipping tables and figures, which are extracted separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bs4 import Tag

from preprocessing.parser_utils import extract_text_from_element


@dataclass
class SectionSchema:
    """Represents a structural section or subsection."""

    title: str
    content: str
    subsections: list[SectionSchema] = field(default_factory=list)


class SectionExtractor:
    """Extracts nested sections from JATS XML."""

    def extract_sections(self, body_tag: Tag | None) -> list[SectionSchema]:
        """
        Extract all top-level sections from the <body> tag.

        Args:
            body_tag: BeautifulSoup <body> tag.

        Returns:
            List of SectionSchema objects.
        """
        if not body_tag:
            return []

        sections = []
        # Find all direct <sec> children of <body>
        for sec in body_tag.find_all("sec", recursive=False):
            extracted = self._process_section(sec)
            if extracted:
                sections.append(extracted)

        return sections

    def _process_section(self, sec_tag: Tag) -> SectionSchema | None:
        """
        Recursively process a <sec> tag.

        Args:
            sec_tag: A <sec> BeautifulSoup tag.

        Returns:
            SectionSchema object or None if empty.
        """
        title_tag = sec_tag.find("title", recursive=False)
        title = extract_text_from_element(title_tag) if title_tag else "Untitled Section"

        content_parts = []
        subsections = []

        # Iterate over all direct children to preserve order
        for child in sec_tag.children:
            if not isinstance(child, Tag):
                continue
            
            tag_name = child.name
            
            if tag_name == "title":
                # Already handled
                continue
            elif tag_name == "sec":
                # Process nested subsection
                subsec = self._process_section(child)
                if subsec:
                    subsections.append(subsec)
            elif tag_name in ["table-wrap", "fig", "disp-formula"]:
                # We skip tables and figures as they are handled by separate extractors.
                # Equations (disp-formula) could be included here or extracted separately.
                # The requirements say "Preserve equations as text", so we will extract
                # them if they have text.
                if tag_name == "disp-formula":
                    eq_text = extract_text_from_element(child)
                    if eq_text:
                        content_parts.append(eq_text)
                continue
            else:
                # Normal paragraph, list, etc.
                text = extract_text_from_element(child)
                if text:
                    content_parts.append(text)

        content = " ".join(content_parts).strip()
        
        if not content and not subsections:
            # If the section has no content and no subsections, skip it
            return None

        return SectionSchema(
            title=title,
            content=content,
            subsections=subsections
        )
