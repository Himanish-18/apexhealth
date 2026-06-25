"""
Healthcare Knowledge Navigator — Chunk Schema.

Defines the core dataclasses and enums that represent the chunk-level
data model. These structures are the output schema for the chunking
pipeline and the direct input to Phase 4 (embedding generation).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ChunkType(str, Enum):
    """Enumeration of supported chunk types for retrieval filtering."""

    TEXT = "TEXT"
    TABLE = "TABLE"
    ABSTRACT = "ABSTRACT"
    FIGURE_CAPTION = "FIGURE_CAPTION"


@dataclass
class ChunkMetadata:
    """Metadata associated with a single chunk.

    Every field supports exact traceability back to the source document,
    section, and subsection.
    """

    pmcid: str
    title: str
    journal: str
    year: int | None
    section: str
    subsection: str
    section_path: list[str]
    chunk_type: str  # ChunkType.value
    token_count: int


@dataclass
class Chunk:
    """A single retrieval unit produced by the chunking pipeline.

    Attributes:
        chunk_id: Unique identifier encoding paper, section, and sequence.
                  Format: {pmcid}_{section_slug}_{seq:03d}
        chunk_type: One of ChunkType values.
        content: The raw text content of this chunk.
        token_count: Number of tokens in content.
        metadata: Full metadata for retrieval and citation lineage.
        metadata_header: Pre-built retrieval header (stored separately,
                         NOT prepended to content).
    """

    chunk_id: str
    chunk_type: str
    content: str
    token_count: int
    metadata: ChunkMetadata
    metadata_header: str


@dataclass
class ChunkedDocument:
    """Container for all chunks derived from a single canonical document.

    Attributes:
        document_metadata: Top-level document metadata dict.
        chunks: Ordered list of chunks extracted from the document.
    """

    document_metadata: dict[str, Any]
    chunks: list[Chunk]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the chunked document to a JSON-compatible dict.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "document_metadata": self.document_metadata,
            "chunks": [asdict(c) for c in self.chunks],
        }


def slugify_section(title: str) -> str:
    """Convert a section title to a URL-safe slug for use in chunk IDs.

    Args:
        title: The section title (e.g., "Statistical Analysis").

    Returns:
        A lowercase slug (e.g., "statistical_analysis").
    """
    if not title:
        return "untitled"
    slug = title.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug or "untitled"
