"""
Tests for index pipeline (indexing.index_pipeline).
Tests resume/force/recreate logic using mocked Qdrant and components.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from indexing.collection_manager import CollectionConfig, CollectionManager
from indexing.index_pipeline import IndexPipeline
from indexing.sparse_index import SparseEncoder


class FakeSparseEncoder(SparseEncoder):
    """Fake SparseEncoder that returns deterministic values."""

    def __init__(self):
        super().__init__()
        self._model = True  # Prevent lazy loading

    def encode(self, texts):
        return [
            models.SparseVector(indices=[1, 2, 3], values=[0.5, 0.3, 0.7])
            for _ in texts
        ]


def _create_embedding_file(path: Path, pmcid: str = "PMC123", n_chunks: int = 2):
    """Write a minimal embedding JSON file."""
    chunks = []
    for i in range(n_chunks):
        vec = np.random.RandomState(42 + i).randn(768).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        chunks.append({
            "chunk_id": f"{pmcid}_sec_{i:03d}",
            "embedding": vec.tolist(),
            "content": f"Medical content {i}.",
            "metadata_header": f"PMCID: {pmcid}",
            "metadata": {
                "pmcid": pmcid,
                "title": "Test",
                "journal": "Nature",
                "year": 2024,
                "section": "Results",
                "subsection": "",
                "section_path": ["Results"],
                "chunk_type": "TEXT",
                "token_count": 50,
            },
        })
    doc = {
        "document_metadata": {"pmcid": pmcid, "title": "Test", "journal": "Nature", "year": 2024},
        "embedding_model": "MedCPT",
        "embedding_dimension": 768,
        "chunks": chunks,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)


def _make_mock_settings(tmp_path: Path):
    """Create mock Settings pointing to tmp directories."""
    embeddings_dir = tmp_path / "data" / "embeddings"
    logs_dir = tmp_path / "data" / "logs" / "indexing"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    settings = MagicMock()
    settings.embeddings_dir = embeddings_dir
    settings.indexing_logs_dir = logs_dir
    settings.qdrant_host = "localhost"
    settings.qdrant_port = 6333
    settings.collection_name = "test_pipeline_collection"
    settings.dense_vector_name = "dense"
    settings.sparse_vector_name = "sparse-bm25"
    settings.dense_vector_size = 768
    settings.indexing_batch_size = 256
    settings.sparse_model_name = "Qdrant/bm25"
    return settings


@pytest.fixture
def in_memory_client():
    """Create an in-memory QdrantClient."""
    return QdrantClient(":memory:")


def test_pipeline_indexes_files(tmp_path, in_memory_client):
    """Pipeline should index embedding files into Qdrant."""
    settings = _make_mock_settings(tmp_path)

    _create_embedding_file(
        settings.embeddings_dir / "PMC100_embeddings.json", "PMC100", n_chunks=3
    )

    pipeline = IndexPipeline(settings)

    with patch("indexing.index_pipeline.QdrantManager") as MockManager, \
         patch("indexing.index_pipeline.SparseEncoder", return_value=FakeSparseEncoder()):
        mock_mgr = MockManager.return_value
        mock_mgr.client = in_memory_client
        mock_mgr.health_check.return_value = True

        stats = pipeline.run(force=True, verbose=False)

    assert stats.documents_indexed == 1
    assert stats.chunks_indexed == 3


def test_pipeline_force_reindexes(tmp_path, in_memory_client):
    """Force mode should re-index all chunks."""
    settings = _make_mock_settings(tmp_path)

    _create_embedding_file(
        settings.embeddings_dir / "PMC200_embeddings.json", "PMC200", n_chunks=2
    )

    pipeline = IndexPipeline(settings)

    with patch("indexing.index_pipeline.QdrantManager") as MockManager, \
         patch("indexing.index_pipeline.SparseEncoder", return_value=FakeSparseEncoder()):
        mock_mgr = MockManager.return_value
        mock_mgr.client = in_memory_client
        mock_mgr.health_check.return_value = True

        # First run
        stats1 = pipeline.run(force=True)
        assert stats1.chunks_indexed == 2

        # Second run with force
        stats2 = pipeline.run(force=True)
        assert stats2.chunks_indexed == 2


def test_pipeline_resume_skips_indexed(tmp_path, in_memory_client):
    """Resume should skip already-indexed chunks."""
    settings = _make_mock_settings(tmp_path)

    _create_embedding_file(
        settings.embeddings_dir / "PMC300_embeddings.json", "PMC300", n_chunks=2
    )

    pipeline = IndexPipeline(settings)

    with patch("indexing.index_pipeline.QdrantManager") as MockManager, \
         patch("indexing.index_pipeline.SparseEncoder", return_value=FakeSparseEncoder()):
        mock_mgr = MockManager.return_value
        mock_mgr.client = in_memory_client
        mock_mgr.health_check.return_value = True

        # First run indexes everything
        stats1 = pipeline.run(force=True)
        assert stats1.chunks_indexed == 2

        # Second run with resume should skip
        stats2 = pipeline.run(resume=True, force=False)
        assert stats2.chunks_skipped == 2


def test_pipeline_no_files(tmp_path, in_memory_client):
    """Pipeline should handle no embedding files gracefully."""
    settings = _make_mock_settings(tmp_path)
    pipeline = IndexPipeline(settings)

    with patch("indexing.index_pipeline.QdrantManager") as MockManager, \
         patch("indexing.index_pipeline.SparseEncoder", return_value=FakeSparseEncoder()):
        mock_mgr = MockManager.return_value
        mock_mgr.client = in_memory_client
        mock_mgr.health_check.return_value = True

        stats = pipeline.run()

    assert stats.documents_indexed == 0
    assert stats.chunks_indexed == 0


def test_pipeline_saves_statistics(tmp_path, in_memory_client):
    """Pipeline should save index_statistics.json after run."""
    settings = _make_mock_settings(tmp_path)

    _create_embedding_file(
        settings.embeddings_dir / "PMC400_embeddings.json", "PMC400", n_chunks=1
    )

    pipeline = IndexPipeline(settings)

    with patch("indexing.index_pipeline.QdrantManager") as MockManager, \
         patch("indexing.index_pipeline.SparseEncoder", return_value=FakeSparseEncoder()):
        mock_mgr = MockManager.return_value
        mock_mgr.client = in_memory_client
        mock_mgr.health_check.return_value = True

        pipeline.run(force=True)

    stats_file = settings.indexing_logs_dir / "index_statistics.json"
    assert stats_file.exists()

    with open(stats_file, "r") as f:
        data = json.load(f)
    assert data["documents_indexed"] == 1
    assert data["chunks_indexed"] == 1


def test_pipeline_qdrant_unreachable(tmp_path):
    """Pipeline should raise ConnectionError when Qdrant is unreachable."""
    settings = _make_mock_settings(tmp_path)
    pipeline = IndexPipeline(settings)

    with patch("indexing.index_pipeline.QdrantManager") as MockManager:
        mock_mgr = MockManager.return_value
        mock_mgr.health_check.return_value = False

        with pytest.raises(ConnectionError):
            pipeline.run()


def test_pipeline_limit(tmp_path, in_memory_client):
    """Pipeline should respect the --limit flag."""
    settings = _make_mock_settings(tmp_path)

    for i in range(5):
        _create_embedding_file(
            settings.embeddings_dir / f"PMC{500+i}_embeddings.json",
            f"PMC{500+i}",
            n_chunks=1,
        )

    pipeline = IndexPipeline(settings)

    with patch("indexing.index_pipeline.QdrantManager") as MockManager, \
         patch("indexing.index_pipeline.SparseEncoder", return_value=FakeSparseEncoder()):
        mock_mgr = MockManager.return_value
        mock_mgr.client = in_memory_client
        mock_mgr.health_check.return_value = True

        stats = pipeline.run(limit=2, force=True)

    assert stats.documents_indexed == 2


def test_pipeline_target_pmcid(tmp_path, in_memory_client):
    """Pipeline should process only the targeted PMCID."""
    settings = _make_mock_settings(tmp_path)

    _create_embedding_file(
        settings.embeddings_dir / "PMC600_embeddings.json", "PMC600", n_chunks=2
    )
    _create_embedding_file(
        settings.embeddings_dir / "PMC601_embeddings.json", "PMC601", n_chunks=2
    )

    pipeline = IndexPipeline(settings)

    with patch("indexing.index_pipeline.QdrantManager") as MockManager, \
         patch("indexing.index_pipeline.SparseEncoder", return_value=FakeSparseEncoder()):
        mock_mgr = MockManager.return_value
        mock_mgr.client = in_memory_client
        mock_mgr.health_check.return_value = True

        stats = pipeline.run(target_pmcid="PMC600", force=True)

    assert stats.documents_indexed == 1
    assert stats.chunks_indexed == 2
