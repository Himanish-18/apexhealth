"""
Tests for index statistics (indexing.index_statistics).
Tests stats accumulation, throughput calculation, and serialization.
"""

import json
import time
from pathlib import Path

import pytest

from indexing.index_statistics import IndexStats, IndexStatsCollector


class TestIndexStats:
    """Tests for the IndexStats dataclass."""

    def test_default_values(self):
        """Default stats should be all zeros."""
        stats = IndexStats()
        assert stats.documents_indexed == 0
        assert stats.chunks_indexed == 0
        assert stats.indexing_time == 0.0
        assert stats.upload_throughput == 0.0


class TestIndexStatsCollector:
    """Tests for the statistics collector."""

    def test_accumulation(self):
        """Should accumulate document and chunk counts."""
        collector = IndexStatsCollector()
        collector.start()
        collector.record_document(chunks_indexed=10, chunks_skipped=2)
        collector.record_document(chunks_indexed=5, chunks_skipped=0)
        stats = collector.finalize()

        assert stats.documents_indexed == 2
        assert stats.chunks_indexed == 15
        assert stats.chunks_skipped == 2

    def test_skip_and_failure_tracking(self):
        """Should track skipped and failed documents separately."""
        collector = IndexStatsCollector()
        collector.start()
        collector.record_skip()
        collector.record_skip()
        collector.record_failure()
        stats = collector.finalize()

        assert stats.documents_skipped == 2
        assert stats.documents_failed == 1

    def test_throughput_calculation(self):
        """Throughput should be chunks / elapsed time."""
        collector = IndexStatsCollector()
        collector.start()
        # Simulate some processing time
        time.sleep(0.05)
        collector.record_document(chunks_indexed=100, chunks_skipped=0)
        stats = collector.finalize()

        assert stats.upload_throughput > 0
        assert stats.indexing_time > 0

    def test_batch_latency(self):
        """Average upload latency should track batch durations."""
        collector = IndexStatsCollector()
        collector.start()
        collector.record_batch(count=10, duration=0.5)
        collector.record_batch(count=10, duration=1.0)
        stats = collector.finalize()

        assert stats.batch_count == 2
        assert stats.average_upload_latency == pytest.approx(0.75, abs=0.01)

    def test_batch_error_tracking(self):
        """Should track batch errors."""
        collector = IndexStatsCollector()
        collector.start()
        collector.record_batch_error()
        collector.record_batch_error()
        stats = collector.finalize()

        assert stats.batch_errors == 2
        assert stats.errors == 2  # errors = batch_errors + documents_failed

    def test_collection_size(self):
        """Should pass through collection_size from finalize."""
        collector = IndexStatsCollector()
        collector.start()
        stats = collector.finalize(collection_size=42)
        assert stats.collection_size == 42

    def test_save_to_file(self, tmp_path):
        """Should save statistics as JSON."""
        collector = IndexStatsCollector()
        collector.start()
        collector.record_document(chunks_indexed=5, chunks_skipped=1)

        stats_file = tmp_path / "stats.json"
        collector.save(stats_file, collection_size=5)

        assert stats_file.exists()
        with open(stats_file, "r") as f:
            data = json.load(f)
        assert data["documents_indexed"] == 1
        assert data["chunks_indexed"] == 5
        assert data["collection_size"] == 5

    def test_empty_collector(self):
        """Finalizing without any records should produce zero stats."""
        collector = IndexStatsCollector()
        collector.start()
        stats = collector.finalize()
        assert stats.documents_indexed == 0
        assert stats.upload_throughput == 0.0
        assert stats.average_upload_latency == 0.0
