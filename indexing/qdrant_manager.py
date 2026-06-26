"""
Healthcare Knowledge Navigator — Qdrant Connection Manager.

Thin wrapper around ``QdrantClient`` that manages connection lifecycle,
health checks, and clean shutdown.  All other indexing components receive
the client instance from this manager.
"""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


class QdrantManager:
    """Manages the Qdrant client connection.

    Attributes:
        host: Qdrant server hostname.
        port: Qdrant REST API port.
    """

    def __init__(self, host: str = "localhost", port: int = 6333) -> None:
        """Initialize connection to Qdrant.

        Args:
            host: Qdrant server hostname.
            port: Qdrant REST API port.
        """
        self.host = host
        self.port = port
        self._client = QdrantClient(host=host, port=port, timeout=30)
        logger.info("QdrantManager initialized → %s:%d", host, port)

    @property
    def client(self) -> QdrantClient:
        """Return the underlying QdrantClient instance."""
        return self._client

    def health_check(self) -> bool:
        """Verify that Qdrant is reachable and healthy.

        Returns:
            True if the server responds, False otherwise.
        """
        try:
            # get_collections is a lightweight call to verify connectivity
            self._client.get_collections()
            logger.debug("Qdrant health check passed.")
            return True
        except Exception as e:
            logger.error("Qdrant health check failed: %s", e)
            return False

    def close(self) -> None:
        """Close the Qdrant client connection."""
        try:
            self._client.close()
            logger.info("QdrantManager connection closed.")
        except Exception as e:
            logger.warning("Error closing QdrantManager: %s", e)

    def __enter__(self) -> "QdrantManager":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
