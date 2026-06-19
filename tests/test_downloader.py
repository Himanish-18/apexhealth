"""
Tests for the resilient downloader (ingestion.downloader).

Tests cover duplicate detection, retry logic, timeout handling,
and atomic file writes. All network calls are mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from configs.settings import Settings
from ingestion.downloader import DownloadResult, Downloader
from ingestion.pmc_api import PMCClient


# ── Sample JATS XML for testing ──────────────────────────────────────
SAMPLE_JATS_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE article PUBLIC "-//NLM//DTD JATS (Z39.96) Journal Archiving
and Interchange DTD v1.2 20190208//EN" "JATS-archivearticle1-mathml3.dtd">
<article article-type="research-article">
  <front>
    <journal-meta>
      <journal-title>Test Journal</journal-title>
    </journal-meta>
    <article-meta>
      <article-id pub-id-type="pmc">9999999</article-id>
      <article-id pub-id-type="doi">10.1234/test.2024</article-id>
      <title-group>
        <article-title>Test Article Title</article-title>
      </title-group>
      <contrib-group>
        <contrib contrib-type="author">
          <name><surname>Smith</surname><given-names>John</given-names></name>
        </contrib>
      </contrib-group>
      <pub-date pub-type="epub">
        <year>2024</year>
      </pub-date>
      <abstract>
        <p>This is a test abstract for validation.</p>
      </abstract>
    </article-meta>
  </front>
  <body>
    <p>Body content.</p>
  </body>
</article>
"""


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    """Create test settings pointing to temporary directories."""
    # Override the path properties by setting env vars that point to tmp
    settings = Settings(
        max_retries=2,
        timeout=5,
        request_delay=0.0,
    )

    # Monkey-patch the path properties to use tmp_path
    type(settings).raw_xml_dir = property(lambda self: tmp_path / "raw_xml")
    type(settings).metadata_dir = property(lambda self: tmp_path / "metadata")
    type(settings).invalid_dir = property(lambda self: tmp_path / "invalid")
    type(settings).ingestion_logs_dir = property(lambda self: tmp_path / "logs")

    return settings


@pytest.fixture
def downloader(tmp_settings: Settings) -> Downloader:
    """Create a Downloader instance for testing."""
    return Downloader(tmp_settings)


@pytest.fixture
def mock_pmc_client() -> MagicMock:
    """Create a mocked PMCClient."""
    client = MagicMock(spec=PMCClient)
    client.fetch_xml.return_value = SAMPLE_JATS_XML
    return client


class TestDuplicateDetection:
    """Tests for duplicate detection logic."""

    def test_duplicate_detection_skips(
        self,
        downloader: Downloader,
        mock_pmc_client: MagicMock,
        tmp_settings: Settings,
    ) -> None:
        """Already-downloaded papers should be skipped."""
        # Create the XML file to simulate a previous download
        xml_dir = tmp_settings.raw_xml_dir
        xml_dir.mkdir(parents=True, exist_ok=True)
        (xml_dir / "PMC9999999.xml").write_text("existing content")

        result = downloader.download_paper("PMC9999999", mock_pmc_client)

        assert result.status == "skipped"
        assert result.pmcid == "PMC9999999"
        # fetch_xml should NOT have been called
        mock_pmc_client.fetch_xml.assert_not_called()

    def test_new_paper_not_skipped(
        self,
        downloader: Downloader,
        mock_pmc_client: MagicMock,
    ) -> None:
        """New papers should not be skipped."""
        result = downloader.download_paper("PMC9999999", mock_pmc_client)

        assert result.status in ("success", "invalid")
        mock_pmc_client.fetch_xml.assert_called_once_with("PMC9999999")


class TestDownloadSuccess:
    """Tests for successful download paths."""

    def test_download_success(
        self,
        downloader: Downloader,
        mock_pmc_client: MagicMock,
        tmp_settings: Settings,
    ) -> None:
        """Successful download should save XML and metadata."""
        result = downloader.download_paper("PMC9999999", mock_pmc_client)

        assert result.status == "success"
        assert result.pmcid == "PMC9999999"
        assert result.elapsed >= 0

        # Verify XML file was created
        xml_path = tmp_settings.raw_xml_dir / "PMC9999999.xml"
        assert xml_path.exists()
        assert "Test Article Title" in xml_path.read_text()

        # Verify metadata file was created
        metadata_path = tmp_settings.metadata_dir / "PMC9999999.json"
        assert metadata_path.exists()

    def test_atomic_write(
        self,
        downloader: Downloader,
        mock_pmc_client: MagicMock,
        tmp_settings: Settings,
    ) -> None:
        """After download, no .tmp files should remain."""
        downloader.download_paper("PMC9999999", mock_pmc_client)

        xml_dir = tmp_settings.raw_xml_dir
        tmp_files = list(xml_dir.glob("*.tmp"))
        assert len(tmp_files) == 0, f"Temporary files remain: {tmp_files}"


class TestRetryLogic:
    """Tests for retry with exponential backoff."""

    def test_retry_on_failure(
        self,
        downloader: Downloader,
        tmp_settings: Settings,
    ) -> None:
        """Retries should be triggered on HTTP errors."""
        mock_client = MagicMock(spec=PMCClient)
        # Fail twice, then succeed
        mock_client.fetch_xml.side_effect = [
            requests.ConnectionError("Connection refused"),
            requests.Timeout("Request timed out"),
            SAMPLE_JATS_XML,
        ]

        result = downloader.download_paper("PMC9999999", mock_client)

        assert result.status == "success"
        assert mock_client.fetch_xml.call_count == 3

    def test_exhausted_retries_fails(
        self,
        downloader: Downloader,
    ) -> None:
        """Exceeding max retries should result in a failed status."""
        mock_client = MagicMock(spec=PMCClient)
        # Fail more times than max_retries (2) + 1 = 3
        mock_client.fetch_xml.side_effect = requests.ConnectionError("Down")

        result = downloader.download_paper("PMC9999999", mock_client)

        assert result.status == "failed"
        assert result.error is not None
        assert "Down" in result.error

    def test_timeout_handled(
        self,
        downloader: Downloader,
    ) -> None:
        """Timeout errors should not crash the pipeline."""
        mock_client = MagicMock(spec=PMCClient)
        mock_client.fetch_xml.side_effect = requests.Timeout("Timed out")

        result = downloader.download_paper("PMC9999999", mock_client)

        assert result.status == "failed"
        assert result.error is not None
        # The pipeline should NOT have raised
