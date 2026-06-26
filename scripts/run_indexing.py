#!/usr/bin/env python3
"""
Healthcare Knowledge Navigator — Indexing Pipeline CLI.

Command-line entry point for indexing biomedical embeddings into Qdrant.
Reads Phase 4 embedding files and builds a hybrid-search-ready collection
with dense + sparse (BM25) vectors and metadata payloads.

Usage::

    python scripts/run_indexing.py
    python scripts/run_indexing.py --limit 100 --verbose
    python scripts/run_indexing.py --pmcid PMC1234567
    python scripts/run_indexing.py --force --batch-size 128
    python scripts/run_indexing.py --recreate
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

# Ensure the project root is on sys.path for imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from configs.settings import get_settings
from indexing.index_pipeline import IndexPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="run_indexing",
        description=(
            "Indexing Pipeline — Ingests biomedical embeddings into a "
            "hybrid-search-ready Qdrant collection with dense + BM25 "
            "sparse vectors and metadata payloads."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of embedding files to index",
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
        help="Force re-indexing of all chunks, overwriting existing points",
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
        default=False,
        help="Drop and recreate the collection before indexing",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of points per upsert batch (default: 256)",
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
        help="Index a single specific document by PMCID (e.g., PMC1234567)",
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
    """Main entry point for the indexing pipeline."""
    args = parse_args()
    setup_root_logging(args.verbose)

    log = logging.getLogger(__name__)

    log.info("=" * 60)
    log.info("Qdrant Indexing Pipeline")
    log.info("=" * 60)
    log.info("Limit:      %s", args.limit or "All")
    log.info("Resume:     %s", args.resume and not args.force)
    log.info("Force:      %s", args.force)
    log.info("Recreate:   %s", args.recreate)
    log.info("Batch Size: %s", args.batch_size or "default (256)")
    log.info("PMCID:      %s", args.pmcid or "All")
    log.info("Verbose:    %s", args.verbose)
    log.info("=" * 60)

    try:
        settings = get_settings()
        pipeline = IndexPipeline(settings)

        stats = pipeline.run(
            limit=args.limit,
            resume=args.resume,
            force=args.force,
            recreate=args.recreate,
            batch_size=args.batch_size,
            verbose=args.verbose,
            target_pmcid=args.pmcid,
        )

        log.info("=" * 60)
        log.info("INDEXING COMPLETE")
        log.info("  Documents Indexed:   %d", stats.documents_indexed)
        log.info("  Documents Skipped:   %d", stats.documents_skipped)
        log.info("  Documents Failed:    %d", stats.documents_failed)
        log.info("  Chunks Indexed:      %d", stats.chunks_indexed)
        log.info("  Chunks Skipped:      %d", stats.chunks_skipped)
        log.info("  Batch Count:         %d", stats.batch_count)
        log.info("  Batch Errors:        %d", stats.batch_errors)
        log.info("  Upload Throughput:   %.2f chunks/sec", stats.upload_throughput)
        log.info("  Avg Upload Latency:  %.4fs", stats.average_upload_latency)
        log.info("  Indexing Time:       %.2fs", stats.indexing_time)
        log.info("  Collection Size:     %d points", stats.collection_size)
        log.info("=" * 60)

        return 0

    except ConnectionError as e:
        log.error("Connection error: %s", e)
        return 1
    except Exception as e:
        log.exception("Indexing pipeline failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
