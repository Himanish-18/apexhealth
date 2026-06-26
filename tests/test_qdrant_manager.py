"""
Tests for Qdrant connection manager (indexing.qdrant_manager).
Tests health check and connection lifecycle using in-memory Qdrant.
"""

from unittest.mock import MagicMock, patch

import pytest

from indexing.qdrant_manager import QdrantManager


class TestQdrantManagerInMemory:
    """Tests using a mocked QdrantClient to avoid real connections."""

    def test_health_check_passes_when_reachable(self):
        """Health check should return True when Qdrant responds."""
        with patch("indexing.qdrant_manager.QdrantClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.get_collections.return_value = MagicMock()

            mgr = QdrantManager(host="localhost", port=6333)
            assert mgr.health_check() is True
            mock_instance.get_collections.assert_called_once()

    def test_health_check_fails_when_unreachable(self):
        """Health check should return False when Qdrant is unreachable."""
        with patch("indexing.qdrant_manager.QdrantClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.get_collections.side_effect = ConnectionError("refused")

            mgr = QdrantManager(host="localhost", port=6333)
            assert mgr.health_check() is False

    def test_close_calls_client_close(self):
        """Close should delegate to the underlying client."""
        with patch("indexing.qdrant_manager.QdrantClient") as MockClient:
            mock_instance = MockClient.return_value

            mgr = QdrantManager(host="localhost", port=6333)
            mgr.close()
            mock_instance.close.assert_called_once()

    def test_context_manager(self):
        """Context manager should close on exit."""
        with patch("indexing.qdrant_manager.QdrantClient") as MockClient:
            mock_instance = MockClient.return_value

            with QdrantManager(host="localhost", port=6333) as mgr:
                assert mgr.client is mock_instance

            mock_instance.close.assert_called_once()

    def test_client_property_returns_instance(self):
        """The client property should return the QdrantClient instance."""
        with patch("indexing.qdrant_manager.QdrantClient") as MockClient:
            mock_instance = MockClient.return_value

            mgr = QdrantManager(host="localhost", port=6333)
            assert mgr.client is mock_instance
