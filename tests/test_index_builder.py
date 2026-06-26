"""
Tests for index builder (indexing.index_builder).
Tests point building, batch upsert, duplicate detection, and retry logic
using in-memory Qdrant.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from indexing.collection_manager import CollectionConfig, CollectionManager
from indexing.index_builder import IndexBuilder, IndexFileResult
from indexing.sparse_index import SparseEncoder


class FakeSparseEmbedding:
    """Fake fastembed sparse embedding."""

    def __init__(self, n_tokens=3):
        self.indices = np.array(list(range(n_tokens)), dtype=np.int64)
        self.values = np.array([0.5] * n_tokens, dtype=np.float32)


class FakeSparseEncoder(SparseEncoder):
    """SparseEncoder that returns deterministic fake embeddings."""

    def __init__(self):
        super().__init__()
        self._model = True  # Prevent lazy loading

    def encode(self, texts):
        return [
            models.SparseVector(
                indices=[100, 200, 300],
                values=[0.5, 0.3, 0.7],
            )
            for _ in texts
        ]


def _create_embedding_file(path: Path, pmcid: str = "PMC123", n_chunks: int = 3):
    """Write a minimal embedding JSON file."""
    chunks = []
    for i in range(n_chunks):
        # Create a fake 768-dim normalized vector
        vec = np.random.RandomState(42 + i).randn(768).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        chunks.append({
            "chunk_id": f"{pmcid}_section_{i:03d}",
            "embedding": vec.tolist(),
            "content": f"Test content for chunk {i} about medical research.",
            "metadata_header": f"PMCID: {pmcid}\nTitle: Test Paper",
            "metadata": {
                "pmcid": pmcid,
                "title": "Test Paper",
                "journal": "Nature",
                "year": 2024,
                "section": "Results",
                "subsection": "",
                "section_path": ["Results"],
                "chunk_type": "TEXT",
                "token_count": 100,
            },
        })

    doc = {
        "document_metadata": {
            "pmcid": pmcid,
            "title": "Test Paper",
            "journal": "Nature",
            "year": 2024,
        },
        "embedding_model": "MedCPT",
        "embedding_dimension": 768,
        "chunks": chunks,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


@pytest.fixture
def client():
    """In-memory Qdrant client."""
    return QdrantClient(":memory:")


@pytest.fixture
def collection(client):
    """Create the test collection."""
    config = CollectionConfig(
        collection_name="test_collection",
        dense_vector_name="dense",
        sparse_vector_name="sparse-bm25",
        dense_vector_size=768,
    )
    mgr = CollectionManager(client, config)
    mgr.create_collection()
    return config


@pytest.fixture
def builder(client, collection):
    """IndexBuilder with fake sparse encoder."""
    return IndexBuilder(
        client=client,
        collection_name=collection.collection_name,
        sparse_encoder=FakeSparseEncoder(),
        dense_vector_name=collection.dense_vector_name,
        sparse_vector_name=collection.sparse_vector_name,
        batch_size=10,
        max_retries=1,
        base_delay=0.01,
    )


class TestBuildPoints:
    """Tests for building PointStruct objects from embedding files."""

    def test_build_points_from_file(self, builder, tmp_path):
        """Should build points from a valid embedding file."""
        emb_file = tmp_path / "PMC100_embeddings.json"
        _create_embedding_file(emb_file, "PMC100", n_chunks=3)

        points = builder.build_points(emb_file)
        assert len(points) == 3
        for p in points:
            assert isinstance(p, models.PointStruct)
            assert isinstance(p.id, int)
            assert p.id > 0

    def test_points_have_dense_and_sparse_vectors(self, builder, tmp_path):
        """Each point should have both dense and sparse vectors."""
        emb_file = tmp_path / "PMC101_embeddings.json"
        _create_embedding_file(emb_file, "PMC101", n_chunks=1)

        points = builder.build_points(emb_file)
        assert len(points) == 1
        assert "dense" in points[0].vector
        assert "sparse-bm25" in points[0].vector

    def test_points_have_payload(self, builder, tmp_path):
        """Each point should have a complete payload."""
        emb_file = tmp_path / "PMC102_embeddings.json"
        _create_embedding_file(emb_file, "PMC102", n_chunks=1)

        points = builder.build_points(emb_file)
        payload = points[0].payload
        assert payload["pmcid"] == "PMC102"
        assert payload["journal"] == "Nature"
        assert payload["year"] == 2024
        assert payload["section"] == "Results"

    def test_empty_chunks_returns_empty(self, builder, tmp_path):
        """File with no chunks should return empty list."""
        emb_file = tmp_path / "PMC103_embeddings.json"
        doc = {
            "document_metadata": {"pmcid": "PMC103"},
            "chunks": [],
        }
        emb_file.parent.mkdir(parents=True, exist_ok=True)
        with open(emb_file, "w") as f:
            json.dump(doc, f)

        points = builder.build_points(emb_file)
        assert points == []

    def test_deterministic_point_ids(self, builder, tmp_path):
        """Same chunk_id should always produce the same point ID."""
        emb_file = tmp_path / "PMC104_embeddings.json"
        _create_embedding_file(emb_file, "PMC104", n_chunks=2)

        points1 = builder.build_points(emb_file)
        points2 = builder.build_points(emb_file)

        assert [p.id for p in points1] == [p.id for p in points2]


class TestUpsertBatch:
    """Tests for batch upsert with retry logic."""

    def test_successful_upsert(self, builder, client, collection, tmp_path):
        """Should upload points successfully."""
        emb_file = tmp_path / "PMC200_embeddings.json"
        _create_embedding_file(emb_file, "PMC200", n_chunks=3)

        points = builder.build_points(emb_file)
        result = builder.upsert_batch(points)
        assert result is True

        # Verify points are in the collection
        info = client.get_collection(collection.collection_name)
        assert info.points_count == 3

    def test_empty_batch_succeeds(self, builder):
        """Empty batch should succeed without errors."""
        assert builder.upsert_batch([]) is True

    def test_retry_on_failure(self, builder):
        """Should retry on transient failures."""
        # Mock client.upsert to fail once then succeed
        original_upsert = builder._client.upsert
        call_count = [0]

        def failing_upsert(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("Transient error")
            return original_upsert(*args, **kwargs)

        builder._client.upsert = failing_upsert

        emb_file = Path(__file__).parent.parent / "tests" / "_fake.json"
        # Create a simple point for testing
        point = models.PointStruct(
            id=12345,
            vector={
                "dense": [0.1] * 768,
                "sparse-bm25": models.SparseVector(
                    indices=[1, 2], values=[0.5, 0.3]
                ),
            },
            payload={"test": True},
        )

        result = builder.upsert_batch([point])
        assert result is True
        assert call_count[0] == 2  # Failed once, succeeded on retry


class TestCheckExisting:
    """Tests for duplicate detection."""

    def test_no_existing_points(self, builder):
        """Should return empty set when no points exist."""
        existing = builder.check_existing([1, 2, 3])
        assert existing == set()

    def test_finds_existing_points(self, builder, client, collection, tmp_path):
        """Should return IDs of already-indexed points."""
        emb_file = tmp_path / "PMC300_embeddings.json"
        _create_embedding_file(emb_file, "PMC300", n_chunks=2)

        points = builder.build_points(emb_file)
        builder.upsert_batch(points)

        existing = builder.check_existing([p.id for p in points])
        assert existing == {p.id for p in points}


class TestIndexFile:
    """Tests for end-to-end file indexing."""

    def test_index_new_file(self, builder, tmp_path):
        """Should index all chunks from a new file."""
        emb_file = tmp_path / "PMC400_embeddings.json"
        _create_embedding_file(emb_file, "PMC400", n_chunks=5)

        result = builder.index_file(emb_file, force=False)
        assert result.pmcid == "PMC400"
        assert result.chunks_indexed == 5
        assert result.chunks_skipped == 0
        assert result.errors == 0

    def test_skip_existing_chunks(self, builder, tmp_path):
        """Should skip already-indexed chunks on re-run."""
        emb_file = tmp_path / "PMC401_embeddings.json"
        _create_embedding_file(emb_file, "PMC401", n_chunks=3)

        # First run indexes everything
        result1 = builder.index_file(emb_file, force=False)
        assert result1.chunks_indexed == 3

        # Second run skips everything
        result2 = builder.index_file(emb_file, force=False)
        assert result2.chunks_indexed == 0
        assert result2.chunks_skipped == 3

    def test_force_reindexes(self, builder, tmp_path):
        """Should re-index all chunks when force=True."""
        emb_file = tmp_path / "PMC402_embeddings.json"
        _create_embedding_file(emb_file, "PMC402", n_chunks=3)

        builder.index_file(emb_file, force=False)
        result = builder.index_file(emb_file, force=True)
        assert result.chunks_indexed == 3
        assert result.chunks_skipped == 0
