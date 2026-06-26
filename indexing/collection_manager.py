"""
Healthcare Knowledge Navigator — Collection Manager.

Creates, deletes, and inspects the ``medical_documents`` Qdrant collection.
Configures dense + sparse (BM25) named vectors and payload indexes for
hybrid retrieval with metadata filtering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)


@dataclass
class CollectionConfig:
    """Configuration for the medical_documents collection.

    Attributes:
        collection_name: Name of the Qdrant collection.
        dense_vector_name: Named vector key for dense embeddings.
        sparse_vector_name: Named vector key for sparse BM25 vectors.
        dense_vector_size: Dimensionality of dense vectors.
    """

    collection_name: str = "medical_documents"
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse-bm25"
    dense_vector_size: int = 768


# Fields that should have payload indexes for filtering.
PAYLOAD_INDEX_FIELDS: list[tuple[str, Any]] = [
    ("pmcid", models.PayloadSchemaType.KEYWORD),
    ("journal", models.PayloadSchemaType.KEYWORD),
    (
        "year",
        models.IntegerIndexParams(
            type=models.IntegerIndexType.INTEGER,
            lookup=True,
            range=True,
        ),
    ),
    ("section", models.PayloadSchemaType.KEYWORD),
    ("chunk_type", models.PayloadSchemaType.KEYWORD),
    ("article_type", models.PayloadSchemaType.KEYWORD),
    ("license", models.PayloadSchemaType.KEYWORD),
]


class CollectionManager:
    """Manages the lifecycle of a hybrid Qdrant collection.

    Creates collections with both dense (cosine) and sparse (BM25 + IDF)
    vector configurations, plus payload indexes for metadata filtering.
    """

    def __init__(
        self,
        client: QdrantClient,
        config: CollectionConfig | None = None,
    ) -> None:
        """Initialize the collection manager.

        Args:
            client: An active QdrantClient instance.
            config: Collection configuration; uses defaults if not provided.
        """
        self._client = client
        self.config = config or CollectionConfig()

    def collection_exists(self) -> bool:
        """Check whether the collection already exists.

        Returns:
            True if the collection exists in Qdrant.
        """
        try:
            return self._client.collection_exists(self.config.collection_name)
        except Exception as e:
            logger.error("Error checking collection existence: %s", e)
            return False

    def create_collection(self) -> bool:
        """Create the hybrid collection if it does not exist.

        Configures:
            - Dense vector: ``VectorParams`` with cosine distance
            - Sparse vector: ``SparseVectorParams`` with IDF modifier
            - Payload indexes for metadata filtering

        Returns:
            True if a new collection was created, False if it already existed.
        """
        if self.collection_exists():
            logger.info(
                "Collection '%s' already exists — skipping creation.",
                self.config.collection_name,
            )
            return False

        logger.info(
            "Creating collection '%s' (dense=%d, sparse=BM25+IDF)...",
            self.config.collection_name,
            self.config.dense_vector_size,
        )

        self._client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config={
                self.config.dense_vector_name: models.VectorParams(
                    size=self.config.dense_vector_size,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                self.config.sparse_vector_name: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                ),
            },
        )

        # Create payload indexes for metadata filtering
        self._create_payload_indexes()

        logger.info("Collection '%s' created successfully.", self.config.collection_name)
        return True

    def delete_collection(self) -> bool:
        """Delete the collection if it exists.

        Returns:
            True if the collection was deleted, False if it didn't exist.
        """
        if not self.collection_exists():
            logger.info(
                "Collection '%s' does not exist — nothing to delete.",
                self.config.collection_name,
            )
            return False

        self._client.delete_collection(self.config.collection_name)
        logger.info("Collection '%s' deleted.", self.config.collection_name)
        return True

    def recreate_collection(self) -> None:
        """Delete and recreate the collection from scratch."""
        logger.info("Recreating collection '%s'...", self.config.collection_name)
        self.delete_collection()
        self.create_collection()

    def get_collection_info(self) -> dict[str, Any]:
        """Retrieve collection metadata and statistics.

        Returns:
            Dict with collection info including point count, vector config,
            and status.  Returns an empty dict if the collection doesn't exist.
        """
        if not self.collection_exists():
            return {}

        try:
            info = self._client.get_collection(self.config.collection_name)
            return {
                "collection_name": self.config.collection_name,
                "status": str(info.status),
                "points_count": info.points_count,
            }
        except Exception as e:
            logger.error("Error getting collection info: %s", e)
            return {}

    def _create_payload_indexes(self) -> None:
        """Create payload indexes for all filterable fields."""
        for field_name, field_schema in PAYLOAD_INDEX_FIELDS:
            try:
                self._client.create_payload_index(
                    collection_name=self.config.collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
                logger.debug("Created payload index: %s", field_name)
            except Exception as e:
                logger.warning(
                    "Failed to create payload index for '%s': %s",
                    field_name,
                    e,
                )
