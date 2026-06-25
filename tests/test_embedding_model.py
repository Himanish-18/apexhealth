"""
Tests for embedding model interface (embeddings.embedding_model).
Tests the abstract contract, mock model behavior, determinism, and normalization.
No real model loading — uses a deterministic fake for speed.
"""

import numpy as np
import pytest
import torch

from embeddings.embedding_model import EmbeddingModel


class FakeEmbeddingModel(EmbeddingModel):
    """Deterministic fake for testing without loading a real model."""

    FAKE_DIM = 128

    @property
    def model_name(self) -> str:
        return "FakeModel"

    @property
    def embedding_dimension(self) -> int:
        return self.FAKE_DIM

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Return deterministic L2-normalized embeddings."""
        if not texts:
            return np.empty((0, self.FAKE_DIM), dtype=np.float32)

        rng = np.random.RandomState(42)
        embeddings = rng.randn(len(texts), self.FAKE_DIM).astype(np.float32)

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        embeddings = embeddings / norms

        return embeddings

    def to_device(self, device: torch.device) -> None:
        pass  # No-op for fake model


def test_abstract_interface_cannot_instantiate():
    """Cannot instantiate EmbeddingModel directly."""
    with pytest.raises(TypeError):
        EmbeddingModel()


def test_mock_model_encode_shape():
    """Fake model should return correct shape (N, dim)."""
    model = FakeEmbeddingModel()
    texts = ["Hello world", "Test sentence", "Another text"]

    result = model.encode(texts)

    assert result.shape == (3, 128)
    assert result.dtype == np.float32


def test_mock_model_empty_input():
    """Empty input should return empty array with correct shape."""
    model = FakeEmbeddingModel()
    result = model.encode([])

    assert result.shape == (0, 128)


def test_mock_model_deterministic():
    """Same input should always produce the same output."""
    model = FakeEmbeddingModel()
    texts = ["Deterministic test input"]

    result1 = model.encode(texts)
    result2 = model.encode(texts)

    np.testing.assert_array_equal(result1, result2)


def test_normalization():
    """All output vectors should have unit L2 norm."""
    model = FakeEmbeddingModel()
    texts = ["Text A", "Text B", "Text C", "Text D"]

    result = model.encode(texts)
    norms = np.linalg.norm(result, axis=1)

    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_model_name_property():
    """model_name property should return the expected string."""
    model = FakeEmbeddingModel()
    assert model.model_name == "FakeModel"


def test_embedding_dimension_property():
    """embedding_dimension property should return the expected int."""
    model = FakeEmbeddingModel()
    assert model.embedding_dimension == 128


def test_single_text_encoding():
    """Encoding a single text should work correctly."""
    model = FakeEmbeddingModel()
    result = model.encode(["Single text"])

    assert result.shape == (1, 128)
    norm = np.linalg.norm(result[0])
    assert abs(norm - 1.0) < 1e-5
