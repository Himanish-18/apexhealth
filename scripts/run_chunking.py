#!/usr/bin/env python3
"""
Healthcare Knowledge Navigator — Chunking Pipeline CLI.

Command-line entry point for running the section-aware semantic chunking
pipeline. Transforms canonical JSON documents into embedding-ready chunks.

Usage::

    python scripts/run_chunking.py
    python scripts/run_chunking.py --limit 100 --verbose
    python scripts/run_chunking.py --pmcid PMC1234567
    python scripts/run_chunking.py --force
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
from preprocessing.chunk_pipeline import ChunkPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="run_chunking",
        description=(
            "Semantic Chunking Pipeline — Converts canonical JSON files "
            "into section-aware, metadata-rich retrieval units."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of files to chunk",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from where the last run stopped (default: True)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-chunking of all files, overwriting existing output",
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
        help="Chunk a single specific document by PMCID (e.g., PMC1234567)",
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
    """Main entry point for the chunking pipeline."""
    args = parse_args()
    setup_root_logging(args.verbose)

    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Semantic Chunking Pipeline")
    logger.info("=" * 60)
    logger.info("Limit:   %s", args.limit or "All")
    logger.info("Resume:  %s", args.resume and not args.force)
    logger.info("Force:   %s", args.force)
    logger.info("PMCID:   %s", args.pmcid or "All")
    logger.info("Verbose: %s", args.verbose)
    logger.info("=" * 60)

    try:
        settings = get_settings()
        pipeline = ChunkPipeline(settings)

        stats = pipeline.run(
            limit=args.limit,
            resume=args.resume,
            force=args.force,
            verbose=args.verbose,
            target_pmcid=args.pmcid,
        )

        logger.info("=" * 60)
        logger.info("CHUNKING COMPLETE")
        logger.info("  Documents:       %d", stats.total_documents)
        logger.info("  Total Chunks:    %d", stats.total_chunks)
        logger.info("  Text Chunks:     %d", stats.text_chunks)
        logger.info("  Table Chunks:    %d", stats.table_chunks)
        logger.info("  Figure Chunks:   %d", stats.figure_chunks)
        logger.info("  Abstract Chunks: %d", stats.abstract_chunks)
        logger.info("  Avg Chunk Size:  %.1f tokens", stats.avg_chunk_size)
        logger.info("  Largest Chunk:   %d tokens", stats.largest_chunk)
        logger.info("  Smallest Chunk:  %d tokens", stats.smallest_chunk)
        logger.info("  Rejected:        %d", stats.rejected_chunks)
        logger.info("  Skipped:         %d", stats.skipped)
        logger.info("  Failed:          %d", stats.failed)
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.exception("Chunking pipeline failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
