"""
Healthcare Knowledge Navigator — PMC Ingestion Pipeline Orchestrator.

Ties together all pipeline components (PMCClient, Downloader,
MetadataExtractor, XMLValidator, IngestionLogger) into a single
runnable pipeline with progress tracking, graceful interruption,
and resume capability.

Usage::

    from configs.settings import get_settings
    from ingestion.download_pmc import PMCIngestionPipeline

    settings = get_settings()
    pipeline = PMCIngestionPipeline(settings)
    stats = pipeline.run(limit=100, verbose=True)
"""

from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path

from tqdm import tqdm

from configs.settings import Settings
from ingestion.downloader import Downloader
from ingestion.logger import IngestionLogger, IngestionStats
from ingestion.pmc_api import PMCClient

logger = logging.getLogger(__name__)


class PMCIngestionPipeline:
    """Orchestrates the full PMC Open Access ingestion workflow.

    Searches PMC for Open Access articles, downloads them as JATS XML,
    extracts metadata, validates the XML, and logs all results with
    progress tracking.

    Args:
        settings: Application settings.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._interrupted = False

        # Ensure all required directories exist
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create all required data directories."""
        dirs = [
            self._settings.raw_xml_dir,
            self._settings.metadata_dir,
            self._settings.ingestion_logs_dir,
            self._settings.invalid_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        limit: int | None = None,
        resume: bool = True,
        year: int | None = None,
        journal: str | None = None,
        verbose: bool = False,
        query: str = "",
    ) -> IngestionStats:
        """Execute the ingestion pipeline.

        Steps:
            1. Search PMC for Open Access articles.
            2. Download each article with progress tracking.
            3. Extract metadata and validate.
            4. Log all events and save statistics.

        Args:
            limit: Maximum number of papers to download. Defaults to
                ``settings.max_downloads``.
            resume: If True, skip already-downloaded papers (default).
            year: Filter results to a specific publication year.
            journal: Filter results to a specific journal name.
            verbose: If True, enable DEBUG-level console logging.
            query: Free-text search query for PMC.

        Returns:
            Final IngestionStats with download counts.
        """
        if limit is None:
            limit = self._settings.max_downloads

        # Initialize components
        ing_logger = IngestionLogger(
            logs_dir=self._settings.ingestion_logs_dir,
            verbose=verbose,
        )

        if not resume:
            ing_logger.reset_stats()

        pmc_client = PMCClient(self._settings)
        downloader = Downloader(self._settings)

        # Register signal handler for graceful interruption
        self._interrupted = False
        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_interrupt)

        try:
            # Step 1: Search PMC
            logger.info(
                "Starting PMC ingestion — limit=%d, year=%s, journal=%s",
                limit, year, journal,
            )
            search_result = pmc_client.search(
                query=query,
                max_results=limit,
                year=year,
                journal=journal,
            )

            if not search_result.pmcids:
                logger.warning("No articles found matching the search criteria.")
                ing_logger.save_statistics()
                return ing_logger.get_stats()

            logger.info(
                "Found %d articles to process (total matching: %d)",
                len(search_result.pmcids),
                search_result.total_count,
            )

            # Step 2: Process each article with progress bar
            stats = ing_logger.get_stats()
            with tqdm(
                total=len(search_result.pmcids),
                desc="Downloading",
                unit="paper",
                bar_format=(
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}] "
                    "Skipped: {postfix[skipped]} "
                    "Failed: {postfix[failed]}"
                ),
                postfix={"skipped": stats.skipped, "failed": stats.failed},
            ) as pbar:
                for pmcid in search_result.pmcids:
                    if self._interrupted:
                        logger.info("Graceful shutdown requested. Saving progress...")
                        break

                    # Download the paper
                    result = downloader.download_paper(pmcid, pmc_client)

                    # Log and update stats
                    ing_logger.log_download(
                        pmcid=result.pmcid,
                        status=result.status,
                        elapsed=result.elapsed,
                        error=result.error,
                    )
                    ing_logger.update_stats(result.status)

                    # Update progress bar postfix
                    current_stats = ing_logger.get_stats()
                    pbar.set_postfix(
                        skipped=current_stats.skipped,
                        failed=current_stats.failed,
                    )
                    pbar.update(1)

            # Step 3: Save final statistics
            ing_logger.save_statistics()
            final_stats = ing_logger.get_stats()

            logger.info(
                "Ingestion complete — Downloaded: %d, Skipped: %d, "
                "Failed: %d, Invalid: %d",
                final_stats.downloaded,
                final_stats.skipped,
                final_stats.failed,
                final_stats.invalid,
            )

            return final_stats

        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_sigint)
            pmc_client.close()

    def _handle_interrupt(self, signum: int, frame: object) -> None:
        """Handle SIGINT for graceful shutdown.

        Sets the interrupted flag so the main loop can finish the
        current download and save progress.

        Args:
            signum: Signal number.
            frame: Current stack frame.
        """
        if self._interrupted:
            # Second Ctrl+C — force exit
            logger.warning("Force shutdown requested.")
            sys.exit(1)

        logger.info(
            "\nInterrupt received — finishing current download. "
            "Press Ctrl+C again to force quit."
        )
        self._interrupted = True
