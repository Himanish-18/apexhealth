"""
Healthcare Knowledge Navigator — Resilient File Downloader.

Core download engine with fault tolerance for fetching PMC JATS XML
files. Supports retry logic with exponential backoff and jitter,
timeout handling, duplicate detection, and atomic file writes.

Design principles:
    - Never crash the pipeline because one paper fails.
    - Atomic writes (tmp → rename) prevent corrupted files.
    - Exponential backoff with jitter prevents thundering-herd on NCBI.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

import requests

from configs.settings import Settings
from ingestion.metadata_extractor import MetadataExtractor, PaperMetadata
from ingestion.pmc_api import PMCClient
from ingestion.validator import ValidationResult, XMLValidator

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class DownloadResult:
    """Result of a single paper download attempt.

    Attributes:
        pmcid: The PMC identifier.
        status: Outcome — "success", "skipped", or "failed".
        error: Error message if the download failed.
        elapsed: Time elapsed for the operation in seconds.
        metadata: Extracted metadata (only on success).
        validation: Validation result (only if validated).
    """

    pmcid: str = ""
    status: str = ""  # "success" | "skipped" | "failed" | "invalid"
    error: str | None = None
    elapsed: float = 0.0
    metadata: PaperMetadata | None = None
    validation: ValidationResult | None = None


class Downloader:
    """Resilient downloader for PMC JATS XML papers.

    Orchestrates the full download lifecycle for each paper:
    duplicate check → fetch XML → atomic save → extract metadata → validate.

    Args:
        settings: Application settings with retry, timeout, and path config.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._save_dir = settings.raw_xml_dir
        self._metadata_dir = settings.metadata_dir
        self._invalid_dir = settings.invalid_dir
        self._max_retries = settings.max_retries
        self._timeout = settings.timeout

        self._validator = XMLValidator()
        self._metadata_extractor = MetadataExtractor()

        # Ensure directories exist
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_dir.mkdir(parents=True, exist_ok=True)
        self._invalid_dir.mkdir(parents=True, exist_ok=True)

    def download_paper(
        self,
        pmcid: str,
        pmc_client: PMCClient,
    ) -> DownloadResult:
        """Download a single paper, handling all failure modes.

        Steps:
            1. Check if already downloaded (duplicate detection).
            2. Fetch XML via PMC API with retries.
            3. Save XML atomically (write to .tmp, then rename).
            4. Extract and save metadata.
            5. Validate the XML.
            6. Quarantine if invalid.

        Args:
            pmcid: The PMC identifier to download.
            pmc_client: An initialized PMCClient instance.

        Returns:
            DownloadResult with the outcome details. Never raises.
        """
        start_time = time.monotonic()

        try:
            # Step 1: Duplicate detection
            if self._check_duplicate(pmcid):
                return DownloadResult(
                    pmcid=pmcid,
                    status="skipped",
                    elapsed=time.monotonic() - start_time,
                )

            # Step 2: Fetch XML with retry
            xml_content = self._retry_with_backoff(
                pmc_client.fetch_xml,
                pmcid,
            )

            # Step 3: Atomic save
            xml_path = self._save_xml(pmcid, xml_content)

            # Step 4: Extract and save metadata
            metadata = self._metadata_extractor.extract(pmcid, xml_content)
            self._metadata_extractor.save(metadata, self._metadata_dir)

            # Step 5: Validate
            validation = self._validator.validate(pmcid, xml_path)

            # Step 6: Quarantine if invalid
            if not validation.is_valid:
                self._validator.quarantine(xml_path, self._invalid_dir)
                # Also remove the metadata file for invalid papers
                metadata_path = self._metadata_dir / f"{pmcid}.json"
                if metadata_path.exists():
                    metadata_path.unlink()

                return DownloadResult(
                    pmcid=pmcid,
                    status="invalid",
                    error="; ".join(validation.errors),
                    elapsed=time.monotonic() - start_time,
                    validation=validation,
                )

            return DownloadResult(
                pmcid=pmcid,
                status="success",
                elapsed=time.monotonic() - start_time,
                metadata=metadata,
                validation=validation,
            )

        except Exception as e:
            logger.error("Failed to download %s: %s", pmcid, e)
            return DownloadResult(
                pmcid=pmcid,
                status="failed",
                error=str(e),
                elapsed=time.monotonic() - start_time,
            )

    def _check_duplicate(self, pmcid: str) -> bool:
        """Check whether a paper has already been downloaded.

        Args:
            pmcid: The PMC identifier to check.

        Returns:
            True if the XML file already exists.
        """
        xml_path = self._save_dir / f"{pmcid}.xml"
        if xml_path.exists():
            logger.debug("Duplicate detected — %s already exists", pmcid)
            return True
        return False

    def _save_xml(self, pmcid: str, xml_content: str) -> Path:
        """Save XML content to disk atomically.

        Writes to a temporary file first, then renames to the final
        path. This prevents corrupted files if the process is
        interrupted during the write.

        Args:
            pmcid: The PMC identifier (used for the filename).
            xml_content: The raw XML string to save.

        Returns:
            Path to the saved XML file.
        """
        final_path = self._save_dir / f"{pmcid}.xml"
        tmp_path = self._save_dir / f"{pmcid}.xml.tmp"

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(xml_content)

            # Atomic rename
            tmp_path.replace(final_path)
            logger.debug("Saved XML to %s (%d bytes)", final_path, len(xml_content))
            return final_path

        except Exception:
            # Clean up temp file on failure
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _retry_with_backoff(
        self,
        func: Callable[..., T],
        *args: object,
    ) -> T:
        """Execute a function with exponential backoff and jitter.

        Retries on ``requests.RequestException`` and ``ValueError``
        up to ``max_retries`` times. The delay between retries follows:
        ``base_delay * 2^attempt + random_jitter``.

        Args:
            func: The callable to execute.
            *args: Arguments to pass to the callable.

        Returns:
            The return value of the callable on success.

        Raises:
            The last exception if all retries are exhausted.
        """
        last_exception: Exception | None = None
        base_delay = 1.0

        for attempt in range(self._max_retries + 1):
            try:
                return func(*args)
            except (requests.RequestException, ValueError, ConnectionError) as e:
                last_exception = e
                if attempt < self._max_retries:
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    logger.warning(
                        "Attempt %d/%d failed for %s: %s. "
                        "Retrying in %.1fs...",
                        attempt + 1,
                        self._max_retries + 1,
                        args[0] if args else "unknown",
                        e,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "All %d attempts exhausted for %s: %s",
                        self._max_retries + 1,
                        args[0] if args else "unknown",
                        e,
                    )

        raise last_exception  # type: ignore[misc]
