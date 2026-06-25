"""
Tests for embedding statistics (embeddings.embedding_statistics).
Tests accumulation, throughput calculation, and JSON serialization.
"""

import json
import time
from pathlib import Path

from embeddings.embedding_statistics import EmbeddingStats, StatisticsCollector


def test_stats_accumulation():
    """Batch times and chunk counts should accumulate correctly."""
    collector = StatisticsCollector(model="FakeModel", device="cpu", dimension=128)
    collector.start()

    # Simulate 3 batches
    collector.start_batch()
    time.sleep(0.01)
    collector.end_batch(chunk_count=10)

    collector.start_batch()
    time.sleep(0.01)
    collector.end_batch(chunk_count=15)

    collector.start_batch()
    time.sleep(0.01)
    collector.end_batch(chunk_count=5)

    collector.record_document(chunk_count=10)
    collector.record_document(chunk_count=15)
    collector.record_document(chunk_count=5)

    stats = collector.finalize()

    assert stats.total_chunks == 30
    assert stats.total_documents == 3
    assert stats.average_batch_time > 0
    assert stats.model == "FakeModel"
    assert stats.device == "cpu"
    assert stats.embedding_dimension == 128


def test_throughput_calculation():
    """Throughput (chunks/sec) should be computed correctly."""
    collector = StatisticsCollector(model="TestModel", device="cpu", dimension=64)
    collector.start()

    collector.start_batch()
    time.sleep(0.02)  # Ensure measurable elapsed time
    collector.end_batch(chunk_count=100)
    collector.record_document(chunk_count=100)

    stats = collector.finalize()

    assert stats.throughput_chunks_per_second > 0
    assert stats.total_chunks == 100


def test_stats_serialization(tmp_path: Path):
    """Statistics should be saved as valid JSON with all required fields."""
    collector = StatisticsCollector(model="MedCPT", device="cuda", dimension=768)
    collector.start()

    collector.start_batch()
    collector.end_batch(chunk_count=32)
    collector.record_document(chunk_count=32)
    collector.record_norms([1.0, 1.0, 0.999])
    collector.record_skip()
    collector.record_failure()

    output_path = tmp_path / "stats.json"
    collector.save(output_path)

    assert output_path.exists()

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_fields = {
        "total_documents",
        "total_chunks",
        "embedding_dimension",
        "average_embedding_norm",
        "average_batch_time",
        "throughput_chunks_per_second",
        "device",
        "model",
        "generation_time",
        "peak_memory_mb",
        "documents_skipped",
        "documents_failed",
    }

    assert required_fields.issubset(set(data.keys()))
    assert data["model"] == "MedCPT"
    assert data["device"] == "cuda"
    assert data["embedding_dimension"] == 768
    assert data["documents_skipped"] == 1
    assert data["documents_failed"] == 1


def test_norm_averaging():
    """Average embedding norm should be calculated correctly."""
    collector = StatisticsCollector(model="Test", device="cpu", dimension=64)
    collector.start()

    collector.record_norms([1.0, 1.0, 1.0])
    collector.record_norms([0.998, 1.002])

    stats = collector.finalize()

    # Average of [1.0, 1.0, 1.0, 0.998, 1.002] = 1.0
    assert abs(stats.average_embedding_norm - 1.0) < 0.001


def test_empty_stats():
    """Finalize with no data should return zeroed stats without errors."""
    collector = StatisticsCollector(model="Test", device="cpu", dimension=64)
    collector.start()

    stats = collector.finalize()

    assert stats.total_documents == 0
    assert stats.total_chunks == 0
    assert stats.average_batch_time == 0.0
    assert stats.throughput_chunks_per_second == 0.0
    assert stats.average_embedding_norm == 0.0


def test_peak_memory_tracked():
    """Peak memory should be a positive value after finalization."""
    collector = StatisticsCollector(model="Test", device="cpu", dimension=64)
    collector.start()

    collector.start_batch()
    collector.end_batch(chunk_count=10)

    stats = collector.finalize()

    # psutil should report some memory usage
    assert stats.peak_memory_mb > 0
