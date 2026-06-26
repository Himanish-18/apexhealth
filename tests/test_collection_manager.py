"""
Tests for collection manager (indexing.collection_manager).
Tests collection CRUD and configuration using in-memory Qdrant.
"""

import pytest
from qdrant_client import QdrantClient, models

from indexing.collection_manager import CollectionConfig, CollectionManager


@pytest.fixture
def client():
    """Create an in-memory QdrantClient for testing."""
    return QdrantClient(":memory:")


@pytest.fixture
def config():
    """Default collection configuration for tests."""
    return CollectionConfig(
        collection_name="test_medical_docs",
        dense_vector_name="dense",
        sparse_vector_name="sparse-bm25",
        dense_vector_size=768,
    )


@pytest.fixture
def manager(client, config):
    """CollectionManager with in-memory client."""
    return CollectionManager(client, config)


class TestCollectionCreation:
    """Tests for creating collections."""

    def test_create_new_collection(self, manager, client, config):
        """Should create a collection that does not yet exist."""
        result = manager.create_collection()
        assert result is True
        assert client.collection_exists(config.collection_name)

    def test_create_existing_collection_is_idempotent(self, manager):
        """Creating an already-existing collection should return False."""
        manager.create_collection()
        result = manager.create_collection()
        assert result is False

    def test_collection_has_dense_vector_config(self, manager, client, config):
        """Created collection should have dense vector config."""
        manager.create_collection()
        info = client.get_collection(config.collection_name)
        assert config.dense_vector_name in info.config.params.vectors

    def test_collection_has_sparse_vector_config(self, manager, client, config):
        """Created collection should have sparse vector config."""
        manager.create_collection()
        info = client.get_collection(config.collection_name)
        assert config.sparse_vector_name in info.config.params.sparse_vectors


class TestCollectionDeletion:
    """Tests for deleting collections."""

    def test_delete_existing_collection(self, manager, client, config):
        """Should delete an existing collection."""
        manager.create_collection()
        result = manager.delete_collection()
        assert result is True
        assert not client.collection_exists(config.collection_name)

    def test_delete_nonexistent_collection(self, manager):
        """Deleting a non-existent collection should return False."""
        result = manager.delete_collection()
        assert result is False


class TestCollectionRecreate:
    """Tests for recreating collections."""

    def test_recreate_drops_and_creates(self, manager, client, config):
        """Recreate should drop and create the collection."""
        manager.create_collection()
        # Upsert a dummy point so we can verify the collection was reset
        client.upsert(
            collection_name=config.collection_name,
            points=[
                models.PointStruct(
                    id=1,
                    vector={config.dense_vector_name: [0.1] * 768},
                    payload={"test": True},
                ),
            ],
        )
        info_before = client.get_collection(config.collection_name)
        assert info_before.points_count == 1

        manager.recreate_collection()
        info_after = client.get_collection(config.collection_name)
        assert info_after.points_count == 0


class TestCollectionInfo:
    """Tests for collection info retrieval."""

    def test_collection_exists_true(self, manager):
        """collection_exists should return True for existing collection."""
        manager.create_collection()
        assert manager.collection_exists() is True

    def test_collection_exists_false(self, manager):
        """collection_exists should return False for missing collection."""
        assert manager.collection_exists() is False

    def test_get_collection_info(self, manager, config):
        """get_collection_info should return metadata dict."""
        manager.create_collection()
        info = manager.get_collection_info()
        assert info["collection_name"] == config.collection_name
        assert "points_count" in info
        assert "status" in info

    def test_get_collection_info_nonexistent(self, manager):
        """get_collection_info should return empty dict for missing collection."""
        info = manager.get_collection_info()
        assert info == {}
