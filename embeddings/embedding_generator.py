"""
Healthcare Knowledge Navigator — Embedding Generator.

Generates dense vector embeddings for a single document's chunks.
Reads a Phase 3 chunk JSON file, encodes all chunks in batches via
the embedding model, normalizes vectors, and returns a fully formed
``EmbeddedDocument`` ready for serialization.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from embeddings.embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)


@dataclass
class EmbeddedChunk:
    """A chunk with its computed embedding vector.

    Attributes:
        chunk_id: Unique chunk identifier.
        embedding: L2-normalized dense vector as a list of floats.
        metadata: Full chunk metadata dict for retrieval lineage.
        metadata_header: Pre-built retrieval header string.
    """

    chunk_id: str
    embedding: list[float]
    metadata: dict[str, Any]
    metadata_header: str


@dataclass
class EmbeddedDocument:
    """Container for all embedded chunks from a single document.

    Attributes:
        document_metadata: Top-level document metadata dict.
        embedding_model: Name of the model used for generation.
        embedding_dimension: Dimensionality of the embedding vectors.
        generated_at: ISO 8601 timestamp of generation.
        chunks: List of embedded chunks with vectors and metadata.
    """

    document_metadata: dict[str, Any]
    embedding_model: str
    embedding_dimension: int
    generated_at: str
    chunks: list[EmbeddedChunk]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dict matching the Phase 4 output schema.
        """
        return {
            "document_metadata": self.document_metadata,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "generated_at": self.generated_at,
            "chunks": [asdict(c) for c in self.chunks],
        }


class EmbeddingGenerator:
    """Generates embeddings for all chunks in a single document.

    Reads a chunk JSON file produced by Phase 3, encodes chunk content
    using the provided embedding model, and produces an ``EmbeddedDocument``
    with all metadata preserved.

    The metadata_header is prepended to the chunk content before encoding
    to enrich the embedding with bibliographic context.
    """

    def __init__(self, model: EmbeddingModel, batch_size: int = 32) -> None:
        """Initialize the embedding generator.

        Args:
            model: An EmbeddingModel implementation (e.g., MedCPTEmbeddingModel).
            batch_size: Number of chunks to encode per forward pass.
        """
        self.model = model
        self.batch_size = batch_size

    def generate(self, chunk_file: Path) -> EmbeddedDocument:
        """Generate embeddings for all chunks in a document file.

        Args:
            chunk_file: Path to a Phase 3 chunk JSON file.

        Returns:
            EmbeddedDocument with all chunks embedded and metadata preserved.

        Raises:
            FileNotFoundError: If the chunk file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        with open(chunk_file, "r", encoding="utf-8") as f:
            doc_data = json.load(f)

        document_metadata = doc_data.get("document_metadata", {})
        raw_chunks = doc_data.get("chunks", [])

        if not raw_chunks:
            logger.warning("No chunks found in %s", chunk_file.name)
            return EmbeddedDocument(
                document_metadata=document_metadata,
                embedding_model=self.model.model_name,
                embedding_dimension=self.model.embedding_dimension,
                generated_at=datetime.now(timezone.utc).isoformat(),
                chunks=[],
            )

        embedded_chunks = self._embed_chunks(raw_chunks)

        return EmbeddedDocument(
            document_metadata=document_metadata,
            embedding_model=self.model.model_name,
            embedding_dimension=self.model.embedding_dimension,
            generated_at=datetime.now(timezone.utc).isoformat(),
            chunks=embedded_chunks,
        )

    def _embed_chunks(self, raw_chunks: list[dict[str, Any]]) -> list[EmbeddedChunk]:
        """Embed all chunks in batches.

        Prepends metadata_header to content for richer embeddings.
        Skips individual chunks that fail, logging the error.

        Args:
            raw_chunks: List of chunk dicts from the chunk JSON.

        Returns:
            List of EmbeddedChunk objects with valid embeddings.
        """
        # Prepare texts: prepend metadata header to content for context-aware embeddings
        texts: list[str] = []
        valid_indices: list[int] = []

        for i, chunk in enumerate(raw_chunks):
            try:
                content = chunk.get("content", "")
                header = chunk.get("metadata_header", "")

                # Combine header and content for embedding
                if header:
                    text = f"{header}\n\n{content}"
                else:
                    text = content

                if not text.strip():
                    logger.warning(
                        "Skipping chunk %s: empty content",
                        chunk.get("chunk_id", f"index_{i}"),
                    )
                    continue

                texts.append(text)
                valid_indices.append(i)
            except Exception as e:
                logger.error(
                    "Error preparing chunk %s: %s",
                    chunk.get("chunk_id", f"index_{i}"),
                    e,
                )

        if not texts:
            return []

        # Encode in batches
        all_embeddings = self.model.encode(texts, batch_size=self.batch_size)

        # Build EmbeddedChunk objects
        embedded: list[EmbeddedChunk] = []
        for idx, orig_idx in enumerate(valid_indices):
            chunk = raw_chunks[orig_idx]
            try:
                embedding_vector = all_embeddings[idx].tolist()

                embedded.append(EmbeddedChunk(
                    chunk_id=chunk.get("chunk_id", ""),
                    embedding=embedding_vector,
                    metadata=chunk.get("metadata", {}),
                    metadata_header=chunk.get("metadata_header", ""),
                ))
            except Exception as e:
                logger.error(
                    "Error creating embedded chunk %s: %s",
                    chunk.get("chunk_id", f"index_{orig_idx}"),
                    e,
                )

        return embedded

    def compute_norms(self, embedded_chunks: list[EmbeddedChunk]) -> list[float]:
        """Compute L2 norms for a list of embedded chunks.

        Args:
            embedded_chunks: List of EmbeddedChunk objects.

        Returns:
            List of L2 norms (should all be ~1.0 for normalized embeddings).
        """
        norms = []
        for chunk in embedded_chunks:
            norm = math.sqrt(sum(v * v for v in chunk.embedding))
            norms.append(norm)
        return norms
