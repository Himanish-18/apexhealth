"""
Healthcare Knowledge Navigator — Index Builder.

Core batch upload engine that reads Phase 4 embedding files, builds
Qdrant ``PointStruct`` objects with dense + sparse vectors and payloads,
and upserts them in configurable batches with exponential backoff retry.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from indexing.payload_builder import build_payload, generate_point_id
from indexing.sparse_index import SparseEncoder

logger = logging.getLogger(__name__)


@dataclass
class IndexFileResult:
    """Result of indexing a single embedding file.

    Attributes:
        pmcid: Document PMCID.
        chunks_indexed: Number of chunks successfully uploaded.
        chunks_skipped: Number of chunks skipped (already existed).
        batches: Number of batch upserts performed.
        errors: Number of batch upsert errors.
        duration: Total time in seconds.
    """

    pmcid: str = ""
    chunks_indexed: int = 0
    chunks_skipped: int = 0
    batches: int = 0
    errors: int = 0
    duration: float = 0.0


class IndexBuilder:
    """Builds and uploads Qdrant points from embedding files.

    Reads embedding JSON files produced by Phase 4, generates sparse
    BM25 vectors, constructs payloads, and upserts batches to Qdrant
    with retry logic.

    Attributes:
        collection_name: Target Qdrant collection.
        batch_size: Number of points per upsert batch.
        max_retries: Maximum retry attempts for failed batches.
        base_delay: Base delay in seconds for exponential backoff.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        sparse_encoder: SparseEncoder,
        dense_vector_name: str = "dense",
        sparse_vector_name: str = "sparse-bm25",
        batch_size: int = 256,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        """Initialize the index builder.

        Args:
            client: Active QdrantClient instance.
            collection_name: Name of the target collection.
            sparse_encoder: SparseEncoder for BM25 vector generation.
            dense_vector_name: Named vector key for dense embeddings.
            sparse_vector_name: Named vector key for sparse embeddings.
            batch_size: Points per upsert batch.
            max_retries: Retry attempts for failed batches.
            base_delay: Base delay (seconds) for exponential backoff.
        """
        self._client = client
        self.collection_name = collection_name
        self.sparse_encoder = sparse_encoder
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.base_delay = base_delay

    def build_points(
        self,
        embedding_file: Path,
    ) -> list[models.PointStruct]:
        """Read an embedding file and build Qdrant PointStruct objects.

        Generates both dense (from file) and sparse (via BM25 model)
        vectors for each chunk.

        Args:
            embedding_file: Path to a Phase 4 embedding JSON file.

        Returns:
            List of fully-constructed PointStruct objects.

        Raises:
            FileNotFoundError: If the embedding file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        with open(embedding_file, "r", encoding="utf-8") as f:
            doc_data = json.load(f)

        document_metadata = doc_data.get("document_metadata", {})
        chunks = doc_data.get("chunks", [])

        if not chunks:
            logger.warning("No chunks in %s", embedding_file.name)
            return []

        # Collect texts for batch BM25 encoding
        texts: list[str] = []
        for chunk in chunks:
            content = chunk.get("content", "")
            if not content:
                # Fall back to metadata content
                content = chunk.get("metadata", {}).get("content", "")
            texts.append(content)

        # Generate sparse vectors in batch
        sparse_vectors = self.sparse_encoder.encode(texts)

        points: list[models.PointStruct] = []
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("chunk_id", "")
            if not chunk_id:
                logger.warning(
                    "Skipping chunk at index %d in %s: no chunk_id",
                    i,
                    embedding_file.name,
                )
                continue

            # Dense vector from Phase 4 output
            dense_vector = chunk.get("embedding", [])
            if not dense_vector:
                logger.warning("Chunk %s has no embedding — skipping.", chunk_id)
                continue

            # Build payload
            payload = build_payload(chunk, document_metadata)

            # Build the point
            point = models.PointStruct(
                id=generate_point_id(chunk_id),
                vector={
                    self.dense_vector_name: dense_vector,
                    self.sparse_vector_name: sparse_vectors[i],
                },
                payload=payload,
            )
            points.append(point)

        return points

    def check_existing(self, point_ids: list[int]) -> set[int]:
        """Check which point IDs already exist in the collection.

        Uses ``retrieve`` to check for existing points.  Returns the set
        of IDs that are already indexed.

        Args:
            point_ids: List of integer point IDs to check.

        Returns:
            Set of point IDs that already exist in the collection.
        """
        if not point_ids:
            return set()

        try:
            existing = self._client.retrieve(
                collection_name=self.collection_name,
                ids=point_ids,
                with_payload=False,
                with_vectors=False,
            )
            return {p.id for p in existing}
        except Exception as e:
            logger.warning("Error checking existing points: %s", e)
            return set()

    def upsert_batch(self, points: list[models.PointStruct]) -> bool:
        """Upload a batch of points with exponential backoff retry.

        Args:
            points: List of PointStruct objects to upsert.

        Returns:
            True if the batch was uploaded successfully, False otherwise.
        """
        if not points:
            return True

        for attempt in range(self.max_retries + 1):
            try:
                self._client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                )
                return True
            except Exception as e:
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(
                        "Batch upsert failed (attempt %d/%d): %s. "
                        "Retrying in %.1fs...",
                        attempt + 1,
                        self.max_retries + 1,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Batch upsert failed after %d attempts: %s",
                        self.max_retries + 1,
                        e,
                    )
                    return False

        return False  # pragma: no cover

    def index_file(
        self,
        embedding_file: Path,
        force: bool = False,
    ) -> IndexFileResult:
        """Index a single embedding file into Qdrant.

        Reads the file, builds points, filters out existing ones
        (unless force=True), and upserts in batches.

        Args:
            embedding_file: Path to a Phase 4 embedding JSON file.
            force: If True, re-index all chunks regardless of existence.

        Returns:
            IndexFileResult with metrics for this file.
        """
        start_time = time.monotonic()
        pmcid = embedding_file.stem.replace("_embeddings", "")
        result = IndexFileResult(pmcid=pmcid)

        try:
            # Build all points
            points = self.build_points(embedding_file)
            if not points:
                logger.info("No points to index for %s", pmcid)
                return result

            # Filter existing points if not forcing
            if not force:
                all_ids = [p.id for p in points]
                existing_ids = self.check_existing(all_ids)
                if existing_ids:
                    original_count = len(points)
                    points = [p for p in points if p.id not in existing_ids]
                    result.chunks_skipped = original_count - len(points)
                    logger.debug(
                        "%s: %d/%d chunks already indexed — skipping.",
                        pmcid,
                        result.chunks_skipped,
                        original_count,
                    )

            if not points:
                logger.info("%s: all chunks already indexed.", pmcid)
                result.duration = time.monotonic() - start_time
                return result

            # Batch upsert
            for i in range(0, len(points), self.batch_size):
                batch = points[i : i + self.batch_size]
                success = self.upsert_batch(batch)
                result.batches += 1

                if success:
                    result.chunks_indexed += len(batch)
                else:
                    result.errors += 1
                    logger.error(
                        "%s: batch %d failed (%d points lost).",
                        pmcid,
                        result.batches,
                        len(batch),
                    )

        except Exception as e:
            result.errors += 1
            logger.error("Failed to index %s: %s", pmcid, e)

        result.duration = time.monotonic() - start_time
        return result
