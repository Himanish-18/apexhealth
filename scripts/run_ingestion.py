#!/usr/bin/env python3
"""
Healthcare Knowledge Navigator — PMC Ingestion CLI.

Command-line entry point for running the PMC Open Access ingestion
pipeline. Supports filtering by year/journal, resume after
interruption, and configurable download limits.

Usage::

    python scripts/run_ingestion.py
    python scripts/run_ingestion.py --limit 500 --year 2024 --verbose
    python scripts/run_ingestion.py --limit 100 --journal "Nature" --resume
    python scripts/run_ingestion.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path for imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from configs.settings import get_settings
from ingestion.download_pmc import PMCIngestionPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    settings = get_settings()

    parser = argparse.ArgumentParser(
        prog="run_ingestion",
        description=(
            "PMC Open Access Ingestion Pipeline — Downloads biomedical "
            "papers in JATS XML format from PubMed Central."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_ingestion.py --limit 100\n"
            "  python scripts/run_ingestion.py --year 2024 --verbose\n"
            "  python scripts/run_ingestion.py --journal Nature --limit 50\n"
            "  python scripts/run_ingestion.py --resume --limit 1000\n"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=settings.max_downloads,
        help=(
            f"Maximum number of papers to download "
            f"(default: {settings.max_downloads})"
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from where the last run stopped (default: True)",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Start a fresh ingestion run, resetting statistics",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level console logging",
    )

    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Filter results to a specific publication year (e.g., 2024)",
    )

    parser.add_argument(
        "--journal",
        type=str,
        default=None,
        help='Filter results to a specific journal name (e.g., "Nature")',
    )

    parser.add_argument(
        "--query",
        type=str,
        default="",
        help='Free-text search query (e.g., "cancer treatment")',
    )

    return parser.parse_args()


def setup_root_logging(verbose: bool) -> None:
    """Configure root logger for the application.

    Args:
        verbose: If True, sets level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    """Main entry point for the ingestion pipeline.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = parse_args()
    setup_root_logging(args.verbose)

    logger = logging.getLogger(__name__)

    # Determine resume mode
    resume = args.resume and not args.no_resume

    logger.info("=" * 60)
    logger.info("PMC Open Access Ingestion Pipeline")
    logger.info("=" * 60)
    logger.info("Limit:   %d", args.limit)
    logger.info("Resume:  %s", resume)
    logger.info("Year:    %s", args.year or "All")
    logger.info("Journal: %s", args.journal or "All")
    logger.info("Query:   %s", args.query or "(none)")
    logger.info("Verbose: %s", args.verbose)
    logger.info("=" * 60)

    try:
        settings = get_settings()
        pipeline = PMCIngestionPipeline(settings)

        stats = pipeline.run(
            limit=args.limit,
            resume=resume,
            year=args.year,
            journal=args.journal,
            verbose=args.verbose,
            query=args.query,
        )

        logger.info("=" * 60)
        logger.info("INGESTION COMPLETE")
        logger.info("  Downloaded: %d", stats.downloaded)
        logger.info("  Skipped:    %d", stats.skipped)
        logger.info("  Failed:     %d", stats.failed)
        logger.info("  Invalid:    %d", stats.invalid)
        logger.info("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.info("\nIngestion interrupted by user.")
        return 1
    except Exception as e:
        logger.exception("Ingestion pipeline failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
