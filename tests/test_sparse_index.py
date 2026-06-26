"""
Tests for sparse BM25 encoder (indexing.sparse_index).
Tests sparse vector generation using a mocked fastembed model.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qdrant_client import models

from indexing.sparse_index import SparseEncoder


class FakeSparseEmbedding:
    """Fake sparse embedding result mimicking fastembed output."""

    def __init__(self, indices, values):
        self.indices = np.array(indices, dtype=np.int64)
        self.values = np.array(values, dtype=np.float32)


class FakeSparseModel:
    """Fake SparseTextEmbedding model for testing."""

    def embed(self, texts):
        """Return deterministic sparse embeddings for each text."""
        for i, text in enumerate(texts):
            # Generate some fake token indices and weights
            n_tokens = min(len(text.split()), 5)
            indices = list(range(100 + i * 10, 100 + i * 10 + n_tokens))
            values = [0.5 + j * 0.1 for j in range(n_tokens)]
            yield FakeSparseEmbedding(indices, values)


@pytest.fixture
def encoder():
    """Create a SparseEncoder with a mocked model."""
    enc = SparseEncoder(model_name="Qdrant/bm25")
    enc._model = FakeSparseModel()
    return enc


class TestSparseEncoder:
    """Tests for SparseEncoder."""

    def test_encode_returns_sparse_vectors(self, encoder):
        """encode() should return a list of SparseVector objects."""
        texts = ["hello world", "medical research paper"]
        results = encoder.encode(texts)
        assert len(results) == 2
        for sv in results:
            assert isinstance(sv, models.SparseVector)
            assert len(sv.indices) > 0
            assert len(sv.values) > 0
            assert len(sv.indices) == len(sv.values)

    def test_encode_empty_list(self, encoder):
        """encode() with empty list should return empty list."""
        results = encoder.encode([])
        assert results == []

    def test_encode_single(self, encoder):
        """encode_single() should return a single SparseVector."""
        result = encoder.encode_single("test sentence")
        assert isinstance(result, models.SparseVector)
        assert len(result.indices) > 0

    def test_indices_are_integers(self, encoder):
        """Sparse vector indices should be integers."""
        results = encoder.encode(["test text"])
        for idx in results[0].indices:
            assert isinstance(idx, (int, np.integer))

    def test_values_are_floats(self, encoder):
        """Sparse vector values should be floats."""
        results = encoder.encode(["test text"])
        for val in results[0].values:
            assert isinstance(val, (float, np.floating))

    def test_lazy_loading(self):
        """Model should not be loaded until first encode call."""
        enc = SparseEncoder(model_name="Qdrant/bm25")
        assert enc._model is None

    def test_model_loaded_on_encode(self):
        """Model should be loaded on first encode call."""
        enc = SparseEncoder(model_name="Qdrant/bm25")
        with patch("fastembed.SparseTextEmbedding") as MockModel:
            mock_instance = FakeSparseModel()
            MockModel.return_value = mock_instance
            enc.encode(["test"])
            MockModel.assert_called_once_with(model_name="Qdrant/bm25")
