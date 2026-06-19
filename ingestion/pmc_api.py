"""
Healthcare Knowledge Navigator — PMC Open Access API Client.

Encapsulates all communication with the NCBI E-utilities API for
searching and fetching PubMed Central Open Access articles in JATS XML format.

API Endpoints:
    - ESearch: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi
    - EFetch:  https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi

Usage Guidelines:
    - Max 3 requests/second without API key, 10 with API key.
    - Always include tool name and email for NCBI identification.
    - Use history server (usehistory=y) for batch retrieval.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from configs.settings import Settings

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL = f"{EUTILS_BASE}/esearch.fcgi"
EFETCH_URL = f"{EUTILS_BASE}/efetch.fcgi"


@dataclass
class PMCSearchResult:
    """Result container for a PMC ESearch query.

    Attributes:
        pmcids: List of PMC IDs returned by the search.
        total_count: Total number of results matching the query.
        web_env: NCBI history server Web Environment string.
        query_key: NCBI history server query key.
    """

    pmcids: list[str] = field(default_factory=list)
    total_count: int = 0
    web_env: str = ""
    query_key: str = ""


class PMCClient:
    """Client for the NCBI E-utilities API targeting PMC Open Access.

    Handles searching for articles via ESearch and fetching full-text
    JATS XML via EFetch, with proper rate limiting and NCBI identification.

    Args:
        settings: Application settings containing API key, email, and
            tool name for NCBI identification.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._session = requests.Session()
        self._last_request_time: float = 0.0

        # Base params included in every NCBI request
        self._base_params: dict[str, str] = {
            "tool": settings.ncbi_tool,
        }
        if settings.ncbi_email:
            self._base_params["email"] = settings.ncbi_email
        if settings.ncbi_api_key:
            self._base_params["api_key"] = settings.ncbi_api_key

    def _rate_limit(self) -> None:
        """Enforce NCBI rate limits between requests.

        Waits to ensure at least ``request_delay`` seconds between
        consecutive requests (default 0.3s ≈ 3 req/s).
        """
        elapsed = time.monotonic() - self._last_request_time
        wait = self._settings.request_delay - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.monotonic()

    def search(
        self,
        query: str = "",
        max_results: int | None = None,
        year: int | None = None,
        journal: str | None = None,
    ) -> PMCSearchResult:
        """Search PMC Open Access for articles matching the given criteria.

        Constructs a query with the ``open access[filter]`` and optional
        year/journal filters, then calls ESearch with history server enabled.

        Args:
            query: Free-text search query (e.g., "cancer treatment").
                If empty, returns all Open Access articles.
            max_results: Maximum number of PMCIDs to retrieve.
                Defaults to ``settings.max_downloads``.
            year: Filter results to a specific publication year.
            journal: Filter results to a specific journal name.

        Returns:
            PMCSearchResult with the list of PMCIDs and history server tokens.

        Raises:
            requests.RequestException: On network or HTTP errors.
        """
        if max_results is None:
            max_results = self._settings.max_downloads

        # Build the search term
        search_term = self._build_search_term(query, year, journal)
        logger.info("PMC search query: %s (max_results=%d)", search_term, max_results)

        params: dict[str, Any] = {
            **self._base_params,
            "db": "pmc",
            "term": search_term,
            "retmode": "json",
            "retmax": max_results,
            "usehistory": "y",
            "sort": "relevance",
        }

        self._rate_limit()
        response = self._session.get(
            ESEARCH_URL,
            params=params,
            timeout=self._settings.timeout,
        )
        response.raise_for_status()

        data = response.json()
        esearch_result = data.get("esearchresult", {})

        pmcids = [
            f"PMC{uid}" if not uid.startswith("PMC") else uid
            for uid in esearch_result.get("idlist", [])
        ]

        result = PMCSearchResult(
            pmcids=pmcids,
            total_count=int(esearch_result.get("count", 0)),
            web_env=esearch_result.get("webenv", ""),
            query_key=esearch_result.get("querykey", ""),
        )

        logger.info(
            "PMC search returned %d IDs (total matching: %d)",
            len(result.pmcids),
            result.total_count,
        )
        return result

    def fetch_xml(self, pmcid: str) -> str:
        """Fetch the full-text JATS XML for a single PMC article.

        Args:
            pmcid: The PMC identifier (e.g., "PMC1234567"). The "PMC"
                prefix is stripped before sending to the API.

        Returns:
            The raw JATS XML content as a string.

        Raises:
            requests.RequestException: On network or HTTP errors.
            ValueError: If the response is empty.
        """
        # Strip the "PMC" prefix — efetch expects numeric IDs
        numeric_id = pmcid.replace("PMC", "")

        params: dict[str, Any] = {
            **self._base_params,
            "db": "pmc",
            "id": numeric_id,
            "rettype": "full",
            "retmode": "xml",
        }

        self._rate_limit()
        response = self._session.get(
            EFETCH_URL,
            params=params,
            timeout=self._settings.timeout,
        )
        response.raise_for_status()

        xml_content = response.text
        if not xml_content or not xml_content.strip():
            raise ValueError(f"Empty XML response for {pmcid}")

        logger.debug("Fetched XML for %s (%d bytes)", pmcid, len(xml_content))
        return xml_content

    @staticmethod
    def _build_search_term(
        query: str = "",
        year: int | None = None,
        journal: str | None = None,
    ) -> str:
        """Build the ESearch query term with Open Access filter.

        Args:
            query: Free-text search query.
            year: Optional publication year filter.
            journal: Optional journal name filter.

        Returns:
            The assembled search term string.
        """
        parts: list[str] = []

        if query:
            parts.append(query)

        # Always restrict to Open Access subset
        parts.append("open access[filter]")

        if year is not None:
            parts.append(f"{year}[pdat]")

        if journal:
            parts.append(f'"{journal}"[journal]')

        return " AND ".join(parts)

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self) -> PMCClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
