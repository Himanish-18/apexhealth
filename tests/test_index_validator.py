"""
Tests for index validator (indexing.index_validator).
Tests validation report, dimension checks, and payload completeness
using in-memory Qdrant.
"""

import pytest
from qdrant_client import QdrantClient, models

from indexing.collection_manager import CollectionConfig, CollectionManager
from indexing.index_validator import IndexValidator, ValidationReport


@pytest.fixture
def client():
    """In-memory Qdrant client."""
    return QdrantClient(":memory:")


@pytest.fixture
def collection_name():
    return "test_validation"


@pytest.fixture
def setup_collection(client, collection_name):
    """Create a collection with dense + sparse vectors."""
    config = CollectionConfig(
        collection_name=collection_name,
        dense_vector_name="dense",
        sparse_vector_name="sparse-bm25",
        dense_vector_size=768,
    )
    mgr = CollectionManager(client, config)
    mgr.create_collection()
    return config


def _upsert_valid_points(client, collection_name, count=5):
    """Insert valid test points."""
    points = []
    for i in range(count):
        points.append(
            models.PointStruct(
                id=i + 1,
                vector={
                    "dense": [0.1] * 768,
                    "sparse-bm25": models.SparseVector(
                        indices=[10, 20], values=[0.5, 0.3]
                    ),
                },
                payload={
                    "chunk_id": f"PMC100_section_{i:03d}",
                    "pmcid": "PMC100",
                    "title": "Test Paper",
                    "section": "Introduction",
                    "chunk_type": "TEXT",
                    "token_count": 100,
                    "content": f"Content {i}",
                },
            )
        )
    client.upsert(collection_name=collection_name, points=points)


class TestValidationReport:
    """Tests for the ValidationReport dataclass."""

    def test_default_is_valid(self):
        """Default report should be valid."""
        report = ValidationReport()
        assert report.is_valid is True
        assert report.dimension_errors == 0
        assert report.payload_errors == 0


class TestIndexValidator:
    """Tests for IndexValidator with in-memory Qdrant."""

    def test_validate_empty_collection(self, client, collection_name, setup_collection):
        """Validating an empty collection should pass (0 points)."""
        validator = IndexValidator(client, collection_name, expected_dimension=768)
        report = validator.validate_collection(expected_count=0)
        assert report.is_valid is True
        assert report.total_points == 0

    def test_validate_correct_count(self, client, collection_name, setup_collection):
        """Should pass when point count matches expected."""
        _upsert_valid_points(client, collection_name, count=5)
        validator = IndexValidator(client, collection_name, expected_dimension=768)
        report = validator.validate_collection(expected_count=5)
        assert report.is_valid is True
        assert report.total_points == 5

    def test_validate_count_mismatch(self, client, collection_name, setup_collection):
        """Should fail when point count doesn't match expected."""
        _upsert_valid_points(client, collection_name, count=3)
        validator = IndexValidator(client, collection_name, expected_dimension=768)
        report = validator.validate_collection(expected_count=5)
        assert report.is_valid is False
        assert report.total_points == 3
        assert report.expected_points == 5

    def test_validate_no_expected_count(self, client, collection_name, setup_collection):
        """Should pass when no expected count is provided."""
        _upsert_valid_points(client, collection_name, count=3)
        validator = IndexValidator(client, collection_name, expected_dimension=768)
        report = validator.validate_collection()
        assert report.total_points == 3

    def test_validate_payload_completeness(self, client, collection_name, setup_collection):
        """Points with complete payloads should pass."""
        _upsert_valid_points(client, collection_name, count=2)
        validator = IndexValidator(client, collection_name, expected_dimension=768)
        report = validator.validate_collection()
        assert report.payload_errors == 0

    def test_detect_incomplete_payload(self, client, collection_name, setup_collection):
        """Points with missing payload fields should be flagged."""
        # Insert a point with incomplete payload
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=999,
                    vector={
                        "dense": [0.1] * 768,
                        "sparse-bm25": models.SparseVector(
                            indices=[1], values=[0.5]
                        ),
                    },
                    payload={"chunk_id": "incomplete"},  # Missing many fields
                ),
            ],
        )
        validator = IndexValidator(client, collection_name, expected_dimension=768)
        report = validator.validate_collection()
        assert report.payload_errors > 0
        assert report.is_valid is False

    def test_validate_generates_messages(self, client, collection_name, setup_collection):
        """Validation should produce human-readable messages."""
        _upsert_valid_points(client, collection_name, count=2)
        validator = IndexValidator(client, collection_name, expected_dimension=768)
        report = validator.validate_collection()
        assert len(report.messages) > 0
