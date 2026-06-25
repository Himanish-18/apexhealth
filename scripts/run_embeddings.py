#!/usr/bin/env python3
"""
Healthcare Knowledge Navigator — Embedding Pipeline CLI.

Command-line entry point for running the biomedical embedding generation
pipeline. Converts Phase 3 chunk files into dense vector representations
using MedCPT.

Usage::

    python scripts/run_embeddings.py
    python scripts/run_embeddings.py --limit 100 --verbose
    python scripts/run_embeddings.py --pmcid PMC1234567
    python scripts/run_embeddings.py --force --batch-size 64 --device cuda
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
from embeddings.embedding_pipeline import EmbeddingPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="run_embeddings",
        description=(
            "Embedding Generation Pipeline — Converts chunked medical "
            "documents into dense biomedical vector representations "
            "using MedCPT for semantic retrieval."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of documents to embed",
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
        help="Force re-embedding of all documents, overwriting existing output",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of chunks per GPU batch (default: 32)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Force a specific compute device (default: auto-detect)",
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
        help="Embed a single specific document by PMCID (e.g., PMC1234567)",
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
    """Main entry point for the embedding pipeline."""
    args = parse_args()
    setup_root_logging(args.verbose)

    log = logging.getLogger(__name__)

    log.info("=" * 60)
    log.info("Embedding Generation Pipeline")
    log.info("=" * 60)
    log.info("Limit:      %s", args.limit or "All")
    log.info("Resume:     %s", args.resume and not args.force)
    log.info("Force:      %s", args.force)
    log.info("Batch Size: %s", args.batch_size or "default (32)")
    log.info("Device:     %s", args.device or "auto-detect")
    log.info("PMCID:      %s", args.pmcid or "All")
    log.info("Verbose:    %s", args.verbose)
    log.info("=" * 60)

    try:
        settings = get_settings()
        pipeline = EmbeddingPipeline(settings)

        stats = pipeline.run(
            limit=args.limit,
            resume=args.resume,
            force=args.force,
            verbose=args.verbose,
            target_pmcid=args.pmcid,
            batch_size=args.batch_size,
            device=args.device,
        )

        log.info("=" * 60)
        log.info("EMBEDDING GENERATION COMPLETE")
        log.info("  Documents:           %d", stats.total_documents)
        log.info("  Total Chunks:        %d", stats.total_chunks)
        log.info("  Dimension:           %d", stats.embedding_dimension)
        log.info("  Avg Embedding Norm:  %.6f", stats.average_embedding_norm)
        log.info("  Avg Batch Time:      %.4fs", stats.average_batch_time)
        log.info("  Throughput:          %.2f chunks/sec", stats.throughput_chunks_per_second)
        log.info("  Device:              %s", stats.device)
        log.info("  Model:               %s", stats.model)
        log.info("  Generation Time:     %.2fs", stats.generation_time)
        log.info("  Peak Memory:         %.2f MB", stats.peak_memory_mb)
        log.info("  Skipped:             %d", stats.documents_skipped)
        log.info("  Failed:              %d", stats.documents_failed)
        log.info("=" * 60)

        return 0

    except Exception as e:
        log.exception("Embedding pipeline failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
