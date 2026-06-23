#!/usr/bin/env python3
"""
Healthcare Knowledge Navigator — PMC XML Parser CLI.

Command-line entry point for running the JATS XML to JSON Canonicalization pipeline.
Supports multiprocessing and selective execution.

Usage::

    python scripts/run_parser.py
    python scripts/run_parser.py --limit 100 --verbose
    python scripts/run_parser.py --pmcid PMC1234567
    python scripts/run_parser.py --no-resume
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
from preprocessing.parser_pipeline import ParserPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="run_parser",
        description=(
            "PMC XML Parsing Pipeline — Converts raw JATS XML files into canonical JSON format."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of files to parse",
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
        help="Start a fresh parsing run, overwriting/re-parsing files",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level console logging",
    )

    parser.add_argument(
        "--pmcid",
        type=str,
        default=None,
        help="Parse a single specific document by PMCID (e.g., PMC1234567)",
    )

    return parser.parse_args()


def setup_root_logging(verbose: bool) -> None:
    """Configure root logger for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    """Main entry point for the parsing pipeline."""
    args = parse_args()
    setup_root_logging(args.verbose)

    logger = logging.getLogger(__name__)

    resume = args.resume and not args.no_resume

    logger.info("=" * 60)
    logger.info("PMC XML Parsing Pipeline")
    logger.info("=" * 60)
    logger.info("Limit:   %s", args.limit or "All")
    logger.info("Resume:  %s", resume)
    logger.info("PMCID:   %s", args.pmcid or "All")
    logger.info("Verbose: %s", args.verbose)
    logger.info("=" * 60)

    try:
        settings = get_settings()
        pipeline = ParserPipeline(settings)

        stats = pipeline.run(
            limit=args.limit,
            resume=resume,
            verbose=args.verbose,
            target_pmcid=args.pmcid,
        )

        logger.info("=" * 60)
        logger.info("PARSING COMPLETE")
        logger.info("  Parsed:  %d", stats.parsed)
        logger.info("  Skipped: %d", stats.skipped)
        logger.info("  Failed:  %d", stats.failed)
        logger.info("  Invalid: %d", stats.invalid)
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.exception("Parsing pipeline failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
