"""
Tests for the PMC API client (ingestion.pmc_api).

All tests use mocked HTTP responses — no network access required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from configs.settings import Settings
from ingestion.pmc_api import PMCClient, PMCSearchResult


@pytest.fixture
def settings() -> Settings:
    """Create test settings with defaults."""
    return Settings(
        ncbi_api_key="test_key",
        ncbi_email="test@example.com",
        ncbi_tool="TestTool",
        max_downloads=100,
        request_delay=0.0,  # No delay in tests
        timeout=5,
    )


@pytest.fixture
def pmc_client(settings: Settings) -> PMCClient:
    """Create a PMCClient instance for testing."""
    return PMCClient(settings)


class TestPMCClientSearch:
    """Tests for PMCClient.search()."""

    def test_search_returns_pmcids(self, pmc_client: PMCClient) -> None:
        """ESearch should return a list of PMCIDs."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "esearchresult": {
                "count": "3",
                "idlist": ["1234567", "2345678", "3456789"],
                "webenv": "WEBENV_TOKEN",
                "querykey": "1",
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(pmc_client._session, "get", return_value=mock_response):
            result = pmc_client.search(query="cancer", max_results=10)

        assert isinstance(result, PMCSearchResult)
        assert len(result.pmcids) == 3
        assert result.pmcids[0] == "PMC1234567"
        assert result.pmcids[1] == "PMC2345678"
        assert result.total_count == 3
        assert result.web_env == "WEBENV_TOKEN"
        assert result.query_key == "1"

    def test_search_with_year_filter(self, pmc_client: PMCClient) -> None:
        """Year filter should be included in the search term."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "esearchresult": {
                "count": "0",
                "idlist": [],
                "webenv": "",
                "querykey": "",
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(pmc_client._session, "get", return_value=mock_response) as mock_get:
            pmc_client.search(query="", max_results=10, year=2024)

        # Verify the year filter was included in the query
        call_args = mock_get.call_args
        params = call_args.kwargs.get("params", call_args[1].get("params", {}))
        assert "2024[pdat]" in params["term"]

    def test_search_with_journal_filter(self, pmc_client: PMCClient) -> None:
        """Journal filter should be included in the search term."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "esearchresult": {
                "count": "0",
                "idlist": [],
                "webenv": "",
                "querykey": "",
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(pmc_client._session, "get", return_value=mock_response) as mock_get:
            pmc_client.search(query="", max_results=10, journal="Nature")

        call_args = mock_get.call_args
        params = call_args.kwargs.get("params", call_args[1].get("params", {}))
        assert '"Nature"[journal]' in params["term"]

    def test_search_empty_result(self, pmc_client: PMCClient) -> None:
        """Empty search results should return an empty list."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "esearchresult": {
                "count": "0",
                "idlist": [],
                "webenv": "",
                "querykey": "",
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(pmc_client._session, "get", return_value=mock_response):
            result = pmc_client.search(query="nonexistent_query_xyz")

        assert result.pmcids == []
        assert result.total_count == 0


class TestPMCClientFetch:
    """Tests for PMCClient.fetch_xml()."""

    def test_fetch_xml_returns_content(self, pmc_client: PMCClient) -> None:
        """EFetch should return the raw XML content."""
        expected_xml = '<?xml version="1.0"?><article><body>Test</body></article>'
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = expected_xml
        mock_response.raise_for_status = MagicMock()

        with patch.object(pmc_client._session, "get", return_value=mock_response):
            xml_content = pmc_client.fetch_xml("PMC1234567")

        assert xml_content == expected_xml

    def test_fetch_xml_strips_pmc_prefix(self, pmc_client: PMCClient) -> None:
        """The PMC prefix should be stripped before sending to the API."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<article>Test</article>"
        mock_response.raise_for_status = MagicMock()

        with patch.object(pmc_client._session, "get", return_value=mock_response) as mock_get:
            pmc_client.fetch_xml("PMC1234567")

        call_args = mock_get.call_args
        params = call_args.kwargs.get("params", call_args[1].get("params", {}))
        assert params["id"] == "1234567"

    def test_fetch_xml_empty_response_raises(self, pmc_client: PMCClient) -> None:
        """Empty XML response should raise ValueError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = ""
        mock_response.raise_for_status = MagicMock()

        with patch.object(pmc_client._session, "get", return_value=mock_response):
            with pytest.raises(ValueError, match="Empty XML response"):
                pmc_client.fetch_xml("PMC1234567")


class TestBuildSearchTerm:
    """Tests for the static _build_search_term helper."""

    def test_basic_query(self) -> None:
        """Should include open access filter."""
        term = PMCClient._build_search_term(query="cancer")
        assert "cancer" in term
        assert "open access[filter]" in term

    def test_year_filter(self) -> None:
        """Year should be appended as [pdat] filter."""
        term = PMCClient._build_search_term(year=2024)
        assert "2024[pdat]" in term

    def test_journal_filter(self) -> None:
        """Journal should be quoted and appended as [journal] filter."""
        term = PMCClient._build_search_term(journal="The Lancet")
        assert '"The Lancet"[journal]' in term

    def test_combined_filters(self) -> None:
        """All filters should be combined with AND."""
        term = PMCClient._build_search_term(
            query="diabetes", year=2023, journal="BMJ"
        )
        assert "diabetes" in term
        assert "open access[filter]" in term
        assert "2023[pdat]" in term
        assert '"BMJ"[journal]' in term
        assert " AND " in term
