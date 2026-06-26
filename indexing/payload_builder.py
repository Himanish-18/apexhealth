"""
Healthcare Knowledge Navigator — Payload Builder.

Constructs Qdrant payload dictionaries from Phase 4 chunk data and
generates deterministic integer point IDs from chunk identifiers
using SHA-256.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Required metadata fields in the payload.
REQUIRED_PAYLOAD_FIELDS: list[str] = [
    "chunk_id",
    "pmcid",
    "title",
    "section",
    "chunk_type",
    "token_count",
]


def generate_point_id(chunk_id: str) -> int:
    """Generate a deterministic 64-bit integer point ID from a chunk ID.

    Uses SHA-256 and truncates to 63 bits (positive signed int64) to
    guarantee idempotent indexing — the same chunk_id always maps to
    the same point ID.

    Args:
        chunk_id: Unique chunk identifier string.

    Returns:
        A positive 63-bit integer suitable for Qdrant point IDs.

    Raises:
        ValueError: If chunk_id is empty.
    """
    if not chunk_id:
        raise ValueError("chunk_id must not be empty")

    digest = hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()
    # Use first 16 hex chars (64 bits), mask to 63 bits for signed int64
    return int(digest[:16], 16) & 0x7FFFFFFFFFFFFFFF


def build_payload(
    chunk_data: dict[str, Any],
    document_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a flat Qdrant payload from chunk data and document metadata.

    Merges fields from the chunk's ``metadata`` dict and the top-level
    ``document_metadata`` to produce a complete, flat payload suitable
    for Qdrant filtering and retrieval.

    Args:
        chunk_data: A single chunk dict from the Phase 4 embedding JSON.
        document_metadata: Top-level document metadata dict (optional,
            used as fallback for fields not in chunk metadata).

    Returns:
        Flat dict with all payload fields for the Qdrant point.
    """
    doc_meta = document_metadata or {}
    chunk_meta = chunk_data.get("metadata", {})

    payload: dict[str, Any] = {
        # Core identifiers
        "chunk_id": chunk_data.get("chunk_id", ""),
        "pmcid": chunk_meta.get("pmcid", doc_meta.get("pmcid", "")),
        "title": chunk_meta.get("title", doc_meta.get("title", "")),
        # Bibliographic
        "journal": chunk_meta.get("journal", doc_meta.get("journal", "")),
        "year": chunk_meta.get("year", doc_meta.get("year", 0)),
        # Section hierarchy
        "section": chunk_meta.get("section", ""),
        "subsection": chunk_meta.get("subsection", ""),
        "section_path": chunk_meta.get("section_path", []),
        # Chunk properties
        "chunk_type": chunk_meta.get("chunk_type", chunk_data.get("chunk_type", "TEXT")),
        "token_count": chunk_meta.get("token_count", chunk_data.get("token_count", 0)),
        # Retrieval context
        "metadata_header": chunk_data.get("metadata_header", ""),
        # Additional metadata (may be absent)
        "article_type": chunk_meta.get("article_type", doc_meta.get("article_type", "")),
        "license": chunk_meta.get("license", doc_meta.get("license", "")),
        # Content for BM25 and display
        "content": chunk_data.get("content", chunk_meta.get("content", "")),
    }

    return payload


def validate_payload(payload: dict[str, Any]) -> list[str]:
    """Check that a payload contains all required fields with non-empty values.

    Args:
        payload: The constructed payload dict.

    Returns:
        List of missing or empty required field names.
    """
    missing: list[str] = []
    for field_name in REQUIRED_PAYLOAD_FIELDS:
        value = payload.get(field_name)
        if value is None or value == "" or value == 0:
            missing.append(field_name)
    return missing
