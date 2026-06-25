"""
Healthcare Knowledge Navigator — Embedding Statistics.

Performance monitoring and corpus-level statistics collection for the
embedding generation pipeline. Tracks per-batch timings, memory usage,
throughput, and produces a final statistics JSON report.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingStats:
    """Aggregate statistics for an embedding generation run.

    Attributes:
        total_documents: Number of documents successfully embedded.
        total_chunks: Total number of chunks embedded.
        embedding_dimension: Dimensionality of the embedding vectors.
        average_embedding_norm: Mean L2 norm of all embeddings (should be ~1.0).
        average_batch_time: Mean time per batch in seconds.
        throughput_chunks_per_second: Overall chunks processed per second.
        device: Device type used for generation.
        model: Name of the embedding model.
        generation_time: Total wall-clock time in seconds.
        peak_memory_mb: Peak process memory usage in MB.
        documents_skipped: Number of documents skipped (already embedded).
        documents_failed: Number of documents that failed.
    """

    total_documents: int = 0
    total_chunks: int = 0
    embedding_dimension: int = 0
    average_embedding_norm: float = 0.0
    average_batch_time: float = 0.0
    throughput_chunks_per_second: float = 0.0
    device: str = ""
    model: str = ""
    generation_time: float = 0.0
    peak_memory_mb: float = 0.0
    documents_skipped: int = 0
    documents_failed: int = 0


class StatisticsCollector:
    """Collects performance metrics during embedding generation.

    Usage::

        collector = StatisticsCollector(model="MedCPT", device="cuda", dimension=768)
        collector.start()

        for batch in batches:
            collector.start_batch()
            # ... embed batch ...
            collector.end_batch(chunk_count=len(batch))

        collector.record_document(chunk_count=50)
        stats = collector.finalize()
        collector.save(output_path)
    """

    def __init__(
        self,
        model: str,
        device: str,
        dimension: int,
    ) -> None:
        """Initialize the statistics collector.

        Args:
            model: Name of the embedding model.
            device: Device type string (e.g., "cuda", "cpu").
            dimension: Embedding vector dimensionality.
        """
        self._model = model
        self._device = device
        self._dimension = dimension

        self._start_time: float = 0.0
        self._batch_start_time: float = 0.0
        self._batch_times: list[float] = []
        self._total_documents: int = 0
        self._total_chunks: int = 0
        self._documents_skipped: int = 0
        self._documents_failed: int = 0
        self._norm_sum: float = 0.0
        self._norm_count: int = 0
        self._peak_memory_mb: float = 0.0

    def start(self) -> None:
        """Mark the start of the embedding run."""
        self._start_time = time.time()
        self._update_peak_memory()

    def start_batch(self) -> None:
        """Mark the start of a batch."""
        self._batch_start_time = time.time()

    def end_batch(self, chunk_count: int) -> None:
        """Mark the end of a batch.

        Args:
            chunk_count: Number of chunks in the completed batch.
        """
        elapsed = time.time() - self._batch_start_time
        self._batch_times.append(elapsed)
        self._total_chunks += chunk_count
        self._update_peak_memory()

    def record_document(self, chunk_count: int) -> None:
        """Record a successfully embedded document.

        Args:
            chunk_count: Number of chunks in the document.
        """
        self._total_documents += 1

    def record_skip(self) -> None:
        """Record a skipped document (already embedded)."""
        self._documents_skipped += 1

    def record_failure(self) -> None:
        """Record a failed document."""
        self._documents_failed += 1

    def record_norms(self, norms: list[float]) -> None:
        """Record embedding norms for averaging.

        Args:
            norms: List of L2 norms from a batch of embeddings.
        """
        self._norm_sum += sum(norms)
        self._norm_count += len(norms)

    def finalize(self) -> EmbeddingStats:
        """Compute final aggregate statistics.

        Returns:
            EmbeddingStats with all metrics computed.
        """
        generation_time = time.time() - self._start_time if self._start_time else 0.0

        avg_batch_time = (
            sum(self._batch_times) / len(self._batch_times)
            if self._batch_times
            else 0.0
        )

        throughput = (
            self._total_chunks / generation_time
            if generation_time > 0
            else 0.0
        )

        avg_norm = (
            self._norm_sum / self._norm_count
            if self._norm_count > 0
            else 0.0
        )

        self._update_peak_memory()

        return EmbeddingStats(
            total_documents=self._total_documents,
            total_chunks=self._total_chunks,
            embedding_dimension=self._dimension,
            average_embedding_norm=round(avg_norm, 6),
            average_batch_time=round(avg_batch_time, 4),
            throughput_chunks_per_second=round(throughput, 2),
            device=self._device,
            model=self._model,
            generation_time=round(generation_time, 2),
            peak_memory_mb=round(self._peak_memory_mb, 2),
            documents_skipped=self._documents_skipped,
            documents_failed=self._documents_failed,
        )

    def save(self, path: Path) -> None:
        """Save statistics to a JSON file.

        Args:
            path: Output file path for the statistics JSON.
        """
        stats = self.finalize()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(stats), indent=2),
            encoding="utf-8",
        )
        logger.info("Embedding statistics saved to %s", path)

    def _update_peak_memory(self) -> None:
        """Update peak memory usage from the current process."""
        try:
            process = psutil.Process()
            mem_mb = process.memory_info().rss / (1024 * 1024)
            if mem_mb > self._peak_memory_mb:
                self._peak_memory_mb = mem_mb
        except Exception:
            pass  # psutil may fail in sandboxed environments
