"""
Healthcare Knowledge Navigator — Metadata Extractor.

Extracts structured metadata from PMC JATS XML files using
BeautifulSoup with the lxml parser. Each paper's metadata is saved
as a standalone JSON file for downstream indexing.

Extracted fields:
    PMCID, DOI, Title, Authors, Journal, Publication Year,
    Keywords, Article Type, License, Download Date.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """Structured metadata for a single PMC article.

    Attributes:
        pmcid: PubMed Central identifier (e.g., "PMC1234567").
        doi: Digital Object Identifier, if available.
        title: Article title.
        authors: List of author names.
        journal: Journal title.
        year: Publication year.
        keywords: List of article keywords.
        article_type: Type of article (e.g., "research-article").
        license: License information.
        download_time: ISO-8601 timestamp of when the article was downloaded.
    """

    pmcid: str = ""
    doi: str = ""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    year: int | None = None
    keywords: list[str] = field(default_factory=list)
    article_type: str = ""
    license: str = ""
    download_time: str = ""


class MetadataExtractor:
    """Extracts and persists metadata from PMC JATS XML content.

    Uses BeautifulSoup with the ``lxml-xml`` parser to navigate the JATS
    document tree and extract bibliographic metadata.

    Example::

        extractor = MetadataExtractor()
        metadata = extractor.extract("PMC1234567", xml_content)
        extractor.save(metadata, Path("data/metadata"))
    """

    def extract(self, pmcid: str, xml_content: str) -> PaperMetadata:
        """Extract metadata from raw JATS XML content.

        Args:
            pmcid: The PMC identifier for the article.
            xml_content: Raw JATS XML string.

        Returns:
            PaperMetadata populated with all available fields.
        """
        soup = BeautifulSoup(xml_content, "lxml-xml")

        metadata = PaperMetadata(
            pmcid=pmcid,
            doi=self._extract_doi(soup),
            title=self._extract_title(soup),
            authors=self._extract_authors(soup),
            journal=self._extract_journal(soup),
            year=self._extract_year(soup),
            keywords=self._extract_keywords(soup),
            article_type=self._extract_article_type(soup),
            license=self._extract_license(soup),
            download_time=datetime.now(timezone.utc).isoformat(),
        )

        logger.debug(
            "Extracted metadata for %s: title=%r, authors=%d, keywords=%d",
            pmcid,
            metadata.title[:60] if metadata.title else "",
            len(metadata.authors),
            len(metadata.keywords),
        )
        return metadata

    def save(self, metadata: PaperMetadata, output_dir: Path) -> Path:
        """Save metadata as a JSON file.

        Args:
            metadata: The extracted paper metadata.
            output_dir: Directory to write the JSON file into.

        Returns:
            Path to the created JSON file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{metadata.pmcid}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(metadata), f, indent=2, ensure_ascii=False)

        logger.debug("Saved metadata to %s", output_path)
        return output_path

    # ── Private Extraction Helpers ───────────────────────────────────

    @staticmethod
    def _extract_doi(soup: BeautifulSoup) -> str:
        """Extract DOI from ``<article-id pub-id-type="doi">``."""
        tag = soup.find("article-id", attrs={"pub-id-type": "doi"})
        return tag.get_text(strip=True) if tag else ""

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str:
        """Extract the article title from ``<article-title>``."""
        tag = soup.find("article-title")
        return tag.get_text(strip=True) if tag else ""

    @staticmethod
    def _extract_authors(soup: BeautifulSoup) -> list[str]:
        """Extract author names from ``<contrib contrib-type="author">``."""
        authors: list[str] = []
        for contrib in soup.find_all("contrib", attrs={"contrib-type": "author"}):
            surname_tag = contrib.find("surname")
            given_tag = contrib.find("given-names")

            surname = surname_tag.get_text(strip=True) if surname_tag else ""
            given = given_tag.get_text(strip=True) if given_tag else ""

            if surname and given:
                authors.append(f"{given} {surname}")
            elif surname:
                authors.append(surname)
            elif given:
                authors.append(given)

        return authors

    @staticmethod
    def _extract_journal(soup: BeautifulSoup) -> str:
        """Extract journal title from ``<journal-title>``."""
        tag = soup.find("journal-title")
        return tag.get_text(strip=True) if tag else ""

    @staticmethod
    def _extract_year(soup: BeautifulSoup) -> int | None:
        """Extract publication year from ``<pub-date>``."""
        # Try epub date first, then ppub, then any pub-date
        for pub_type in ["epub", "ppub", None]:
            if pub_type:
                pub_date = soup.find("pub-date", attrs={"pub-type": pub_type})
            else:
                pub_date = soup.find("pub-date")

            if pub_date:
                year_tag = pub_date.find("year")
                if year_tag:
                    try:
                        return int(year_tag.get_text(strip=True))
                    except ValueError:
                        continue

        return None

    @staticmethod
    def _extract_keywords(soup: BeautifulSoup) -> list[str]:
        """Extract keywords from ``<kwd>`` tags."""
        return [
            kwd.get_text(strip=True)
            for kwd in soup.find_all("kwd")
            if kwd.get_text(strip=True)
        ]

    @staticmethod
    def _extract_article_type(soup: BeautifulSoup) -> str:
        """Extract article type from the ``<article>`` tag's ``article-type`` attribute."""
        article_tag = soup.find("article")
        if article_tag and article_tag.get("article-type"):
            return article_tag["article-type"]
        return ""

    @staticmethod
    def _extract_license(soup: BeautifulSoup) -> str:
        """Extract license information from ``<license>``."""
        license_tag = soup.find("license")
        if not license_tag:
            return ""

        # Try xlink:href attribute first (e.g., Creative Commons URL)
        href = license_tag.get("xlink:href", "")
        if href:
            return href

        # Fall back to license text content
        license_p = license_tag.find("license-p")
        if license_p:
            return license_p.get_text(strip=True)[:200]

        return license_tag.get_text(strip=True)[:200]
