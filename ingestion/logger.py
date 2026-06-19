"""
Healthcare Knowledge Navigator — Ingestion Logger.

Provides structured logging for the ingestion pipeline, with separate
file handlers for download events and errors, plus a statistics tracker
that persists aggregate counts to ``statistics.json``.

Log files:
    - ``download.log`` — all download events (success, skip, fail).
    - ``error.log`` — errors only.
    - ``statistics.json`` — aggregate download statistics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    """Aggregate statistics for an ingestion run.

    Attributes:
        downloaded: Number of successfully downloaded papers.
        skipped: Number of papers skipped (already exist).
        failed: Number of papers that failed to download.
        invalid: Number of papers that failed validation.
    """

    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    invalid: int = 0

    @property
    def total_processed(self) -> int:
        """Total number of papers processed (all categories)."""
        return self.downloaded + self.skipped + self.failed + self.invalid


class IngestionLogger:
    """Structured logger for the PMC ingestion pipeline.

    Sets up Python logging with dedicated file handlers for download
    events and errors. Maintains running statistics that can be
    persisted to disk.

    Args:
        logs_dir: Directory to write log files into.
        verbose: If True, sets console logging to DEBUG level.
    """

    def __init__(self, logs_dir: Path, verbose: bool = False) -> None:
        self._logs_dir = logs_dir
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        self._stats = IngestionStats()

        # Load existing statistics if resuming
        self._stats_path = self._logs_dir / "statistics.json"
        self._load_existing_stats()

        # Set up file handlers
        self._setup_logging(verbose)

    def _load_existing_stats(self) -> None:
        """Load existing statistics from disk for resume capability."""
        if self._stats_path.exists():
            try:
                with open(self._stats_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._stats = IngestionStats(**data)
                logger.info(
                    "Loaded existing statistics: %s", asdict(self._stats)
                )
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                logger.warning("Could not load existing statistics: %s", e)

    def _setup_logging(self, verbose: bool) -> None:
        """Configure logging with file handlers and console output.

        Args:
            verbose: If True, console logging is set to DEBUG level.
        """
        # Create the ingestion-specific logger
        ingestion_logger = logging.getLogger("ingestion")
        ingestion_logger.setLevel(logging.DEBUG)

        # Avoid adding duplicate handlers on re-initialization
        if ingestion_logger.handlers:
            return

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Download log — INFO and above
        download_handler = logging.FileHandler(
            self._logs_dir / "download.log",
            encoding="utf-8",
        )
        download_handler.setLevel(logging.INFO)
        download_handler.setFormatter(formatter)
        ingestion_logger.addHandler(download_handler)

        # Error log — WARNING and above
        error_handler = logging.FileHandler(
            self._logs_dir / "error.log",
            encoding="utf-8",
        )
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(formatter)
        ingestion_logger.addHandler(error_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        console_handler.setFormatter(formatter)
        ingestion_logger.addHandler(console_handler)

    def log_download(
        self,
        pmcid: str,
        status: str,
        elapsed: float = 0.0,
        error: str | None = None,
    ) -> None:
        """Log a download event.

        Args:
            pmcid: The PMC identifier.
            status: Download status — "success", "skipped", or "failed".
            elapsed: Time elapsed for the download in seconds.
            error: Error message if the download failed.
        """
        ingestion_logger = logging.getLogger("ingestion")

        if status == "success":
            ingestion_logger.info(
                "DOWNLOADED %s (%.2fs)", pmcid, elapsed
            )
        elif status == "skipped":
            ingestion_logger.info(
                "SKIPPED %s — Already exists.", pmcid
            )
        elif status == "failed":
            ingestion_logger.error(
                "FAILED %s — %s (%.2fs)", pmcid, error or "Unknown error", elapsed
            )
        elif status == "invalid":
            ingestion_logger.warning(
                "INVALID %s — %s", pmcid, error or "Validation failed"
            )

    def log_error(self, pmcid: str, error: str) -> None:
        """Log an error event.

        Args:
            pmcid: The PMC identifier.
            error: Error description.
        """
        ingestion_logger = logging.getLogger("ingestion")
        ingestion_logger.error("ERROR %s — %s", pmcid, error)

    def update_stats(self, status: str) -> None:
        """Increment the appropriate statistics counter.

        Args:
            status: The result status — "success", "skipped", "failed",
                or "invalid".
        """
        if status == "success":
            self._stats.downloaded += 1
        elif status == "skipped":
            self._stats.skipped += 1
        elif status == "failed":
            self._stats.failed += 1
        elif status == "invalid":
            self._stats.invalid += 1

    def save_statistics(self) -> Path:
        """Persist current statistics to ``statistics.json``.

        Returns:
            Path to the saved statistics file.
        """
        with open(self._stats_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self._stats), f, indent=2)

        logger.info("Saved statistics to %s: %s", self._stats_path, asdict(self._stats))
        return self._stats_path

    def get_stats(self) -> IngestionStats:
        """Return the current ingestion statistics.

        Returns:
            A copy of the current IngestionStats.
        """
        return IngestionStats(
            downloaded=self._stats.downloaded,
            skipped=self._stats.skipped,
            failed=self._stats.failed,
            invalid=self._stats.invalid,
        )

    def reset_stats(self) -> None:
        """Reset all statistics counters to zero."""
        self._stats = IngestionStats()
