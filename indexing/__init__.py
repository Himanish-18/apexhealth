"""
Healthcare Knowledge Navigator — Indexing Module.

Handles creating and managing Qdrant collections, upserting
vector embeddings with sparse BM25 vectors, and maintaining the
hybrid search index.

Components:
    - QdrantManager: Connection lifecycle
    - CollectionManager: Collection CRUD
    - PayloadBuilder: Payload construction + deterministic point IDs
    - SparseEncoder: BM25 sparse vector generation
    - IndexBuilder: Batch upload engine
    - IndexValidator: Post-indexing validation
    - IndexStatsCollector: Metrics collection
    - IndexPipeline: Top-level orchestrator

Imports are lazy to avoid requiring qdrant-client at import time
when only lightweight components (payload_builder) are needed.
"""

from indexing.index_statistics import IndexStats, IndexStatsCollector
from indexing.payload_builder import build_payload, generate_point_id, validate_payload

__all__ = [
    # Connection
    "QdrantManager",
    # Collection
    "CollectionManager",
    "CollectionConfig",
    # Payload
    "build_payload",
    "generate_point_id",
    "validate_payload",
    # Sparse encoding
    "SparseEncoder",
    # Index builder
    "IndexBuilder",
    "IndexFileResult",
    # Validation
    "IndexValidator",
    "ValidationReport",
    # Statistics
    "IndexStatsCollector",
    "IndexStats",
    # Pipeline
    "IndexPipeline",
]


def __getattr__(name: str):
    """Lazy imports for heavy dependencies (qdrant-client, fastembed)."""
    if name == "QdrantManager":
        from indexing.qdrant_manager import QdrantManager
        return QdrantManager

    if name in ("CollectionManager", "CollectionConfig"):
        from indexing.collection_manager import CollectionConfig, CollectionManager
        return {"CollectionManager": CollectionManager, "CollectionConfig": CollectionConfig}[name]

    if name == "SparseEncoder":
        from indexing.sparse_index import SparseEncoder
        return SparseEncoder

    if name in ("IndexBuilder", "IndexFileResult"):
        from indexing.index_builder import IndexBuilder, IndexFileResult
        return {"IndexBuilder": IndexBuilder, "IndexFileResult": IndexFileResult}[name]

    if name in ("IndexValidator", "ValidationReport"):
        from indexing.index_validator import IndexValidator, ValidationReport
        return {"IndexValidator": IndexValidator, "ValidationReport": ValidationReport}[name]

    if name == "IndexPipeline":
        from indexing.index_pipeline import IndexPipeline
        return IndexPipeline

    raise AttributeError(f"module 'indexing' has no attribute {name!r}")
