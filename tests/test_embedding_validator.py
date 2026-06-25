"""
Tests for embedding validator (embeddings.embedding_validator).
Tests dimension checks, NaN/Inf detection, unit norm, and empty embeddings.
"""

import math

import pytest

from embeddings.embedding_validator import EmbeddingValidator


def _make_unit_vector(dim: int) -> list[float]:
    """Create a unit vector of the given dimension."""
    import numpy as np
    rng = np.random.RandomState(42)
    v = rng.randn(dim).astype(np.float64)
    v = v / np.linalg.norm(v)
    return v.tolist()


def _make_chunk(chunk_id: str, embedding: list[float]) -> dict:
    """Build a minimal embedded chunk dict for testing."""
    return {
        "chunk_id": chunk_id,
        "embedding": embedding,
        "metadata": {"pmcid": "PMC_TEST"},
        "metadata_header": "PMCID: PMC_TEST",
    }


class TestValidateEmbedding:
    """Tests for single embedding validation."""

    def test_valid_embedding_passes(self):
        """A correct, normalized embedding should pass all checks."""
        validator = EmbeddingValidator(expected_dimension=128)
        embedding = _make_unit_vector(128)

        issues = validator.validate_embedding("chunk_1", embedding)
        assert issues == []

    def test_nan_rejected(self):
        """An embedding containing NaN should be flagged."""
        validator = EmbeddingValidator(expected_dimension=4)
        embedding = [0.5, float("nan"), 0.5, 0.5]

        issues = validator.validate_embedding("chunk_nan", embedding)
        assert any("nan" in issue for issue in issues)

    def test_inf_rejected(self):
        """An embedding containing Inf should be flagged."""
        validator = EmbeddingValidator(expected_dimension=4)
        embedding = [0.5, float("inf"), 0.5, 0.5]

        issues = validator.validate_embedding("chunk_inf", embedding)
        assert any("inf" in issue for issue in issues)

    def test_negative_inf_rejected(self):
        """An embedding containing -Inf should be flagged."""
        validator = EmbeddingValidator(expected_dimension=4)
        embedding = [0.5, float("-inf"), 0.5, 0.5]

        issues = validator.validate_embedding("chunk_neginf", embedding)
        assert any("inf" in issue for issue in issues)

    def test_wrong_dimension_rejected(self):
        """Dimension mismatch should be detected."""
        validator = EmbeddingValidator(expected_dimension=768)
        embedding = _make_unit_vector(128)  # Wrong dimension

        issues = validator.validate_embedding("chunk_dim", embedding)
        assert any("wrong_dimension" in issue for issue in issues)

    def test_non_unit_norm_rejected(self):
        """An unnormalized vector should be flagged."""
        validator = EmbeddingValidator(expected_dimension=4)
        embedding = [1.0, 2.0, 3.0, 4.0]  # norm ≈ 5.48, not 1.0

        issues = validator.validate_embedding("chunk_norm", embedding)
        assert any("non_unit_norm" in issue for issue in issues)

    def test_empty_embedding_rejected(self):
        """An empty embedding should be flagged."""
        validator = EmbeddingValidator(expected_dimension=128)

        issues = validator.validate_embedding("chunk_empty", [])
        assert any("empty" in issue for issue in issues)


class TestValidateDocument:
    """Tests for document-level validation."""

    def test_all_valid(self):
        """All valid embeddings should pass and appear in output."""
        validator = EmbeddingValidator(expected_dimension=128)
        chunks = [
            _make_chunk("c1", _make_unit_vector(128)),
            _make_chunk("c2", _make_unit_vector(128)),
            _make_chunk("c3", _make_unit_vector(128)),
        ]

        valid, report = validator.validate_document(chunks)

        assert len(valid) == 3
        assert report.total_checked == 3
        assert report.total_valid == 3
        assert report.total_invalid == 0
        assert report.issues == []

    def test_mixed_valid_invalid(self):
        """Invalid embeddings should be excluded; valid ones kept."""
        validator = EmbeddingValidator(expected_dimension=128)
        chunks = [
            _make_chunk("c1", _make_unit_vector(128)),      # valid
            _make_chunk("c2", [float("nan")] * 128),         # invalid: NaN
            _make_chunk("c3", _make_unit_vector(128)),       # valid
        ]

        valid, report = validator.validate_document(chunks)

        assert len(valid) == 2
        assert report.total_valid == 2
        assert report.total_invalid == 1
        assert len(report.issues) == 1
        assert report.issues[0]["chunk_id"] == "c2"

    def test_all_invalid(self):
        """All invalid embeddings should result in empty output."""
        validator = EmbeddingValidator(expected_dimension=128)
        chunks = [
            _make_chunk("c1", []),                           # empty
            _make_chunk("c2", _make_unit_vector(64)),        # wrong dim
        ]

        valid, report = validator.validate_document(chunks)

        assert len(valid) == 0
        assert report.total_invalid == 2

    def test_empty_document(self):
        """Empty document should produce empty valid list and clean report."""
        validator = EmbeddingValidator(expected_dimension=128)

        valid, report = validator.validate_document([])

        assert len(valid) == 0
        assert report.total_checked == 0
        assert report.total_valid == 0
        assert report.total_invalid == 0
