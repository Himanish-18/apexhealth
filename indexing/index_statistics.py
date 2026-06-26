"""
Healthcare Knowledge Navigator — Index Statistics.

Collects, aggregates, and serializes indexing pipeline metrics.
Mirrors the ``StatisticsCollector`` pattern from Phase 4 for consistency.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class IndexStats:
    """Aggregate statistics for an indexing run.

    Attributes:
        documents_indexed: Number of embedding files successfully processed.
        documents_skipped: Number of files skipped (all chunks already indexed).
        documents_failed: Number of files that encountered errors.
        chunks_indexed: Total chunks uploaded to Qdrant.
        chunks_skipped: Total chunks skipped (already existed).
        batch_count: Total number of batch upserts.
        batch_errors: Total number of failed batch upserts.
        upload_throughput: Chunks uploaded per second.
        indexing_time: Total wall-clock time in seconds.
        collection_size: Final point count in the collection.
        average_upload_latency: Average time per batch in seconds.
        errors: Total error count.
    """

    documents_indexed: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    chunks_indexed: int = 0
    chunks_skipped: int = 0
    batch_count: int = 0
    batch_errors: int = 0
    upload_throughput: float = 0.0
    indexing_time: float = 0.0
    collection_size: int = 0
    average_upload_latency: float = 0.0
    errors: int = 0


class IndexStatsCollector:
    """Collects metrics during the indexing pipeline run.

    Call ``start()`` at the beginning, ``record_*`` methods during
    processing, and ``finalize()`` at the end to compute derived metrics.
    """

    def __init__(self) -> None:
        """Initialize the statistics collector."""
        self._start_time: float = 0.0
        self._documents_indexed: int = 0
        self._documents_skipped: int = 0
        self._documents_failed: int = 0
        self._chunks_indexed: int = 0
        self._chunks_skipped: int = 0
        self._batch_count: int = 0
        self._batch_errors: int = 0
        self._batch_durations: list[float] = []

    def start(self) -> None:
        """Mark the start of the indexing run."""
        self._start_time = time.monotonic()

    def record_document(self, chunks_indexed: int, chunks_skipped: int) -> None:
        """Record completion of a single document.

        Args:
            chunks_indexed: Chunks successfully uploaded.
            chunks_skipped: Chunks skipped (already existed).
        """
        self._documents_indexed += 1
        self._chunks_indexed += chunks_indexed
        self._chunks_skipped += chunks_skipped

    def record_skip(self) -> None:
        """Record a fully-skipped document (all chunks already indexed)."""
        self._documents_skipped += 1

    def record_failure(self) -> None:
        """Record a document that failed to process."""
        self._documents_failed += 1

    def record_batch(self, count: int, duration: float) -> None:
        """Record a batch upsert.

        Args:
            count: Number of points in the batch.
            duration: Time taken for this batch in seconds.
        """
        self._batch_count += 1
        self._batch_durations.append(duration)

    def record_batch_error(self) -> None:
        """Record a failed batch upsert."""
        self._batch_errors += 1

    def finalize(self, collection_size: int = 0) -> IndexStats:
        """Compute derived metrics and return the final stats.

        Args:
            collection_size: Final point count in the collection.

        Returns:
            IndexStats with all computed fields.
        """
        elapsed = time.monotonic() - self._start_time if self._start_time else 0.0

        throughput = (
            self._chunks_indexed / elapsed if elapsed > 0 else 0.0
        )

        avg_latency = (
            sum(self._batch_durations) / len(self._batch_durations)
            if self._batch_durations
            else 0.0
        )

        return IndexStats(
            documents_indexed=self._documents_indexed,
            documents_skipped=self._documents_skipped,
            documents_failed=self._documents_failed,
            chunks_indexed=self._chunks_indexed,
            chunks_skipped=self._chunks_skipped,
            batch_count=self._batch_count,
            batch_errors=self._batch_errors,
            upload_throughput=round(throughput, 2),
            indexing_time=round(elapsed, 2),
            collection_size=collection_size,
            average_upload_latency=round(avg_latency, 4),
            errors=self._batch_errors + self._documents_failed,
        )

    def save(self, path: Path, collection_size: int = 0) -> None:
        """Finalize and save statistics to a JSON file.

        Args:
            path: Output file path.
            collection_size: Final collection point count.
        """
        stats = self.finalize(collection_size)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(stats), f, indent=2)
        logger.info("Index statistics saved to %s", path)
