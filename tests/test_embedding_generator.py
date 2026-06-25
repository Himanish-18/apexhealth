"""
Tests for embedding generator (embeddings.embedding_generator).
Tests document generation, metadata preservation, output schema, and error handling.
Uses a deterministic fake model — no real model loading.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from embeddings.embedding_generator import EmbeddedDocument, EmbeddingGenerator
from embeddings.embedding_model import EmbeddingModel


class FakeEmbeddingModel(EmbeddingModel):
    """Deterministic fake for testing the generator."""

    FAKE_DIM = 64

    @property
    def model_name(self) -> str:
        return "FakeModel"

    @property
    def embedding_dimension(self) -> int:
        return self.FAKE_DIM

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, self.FAKE_DIM), dtype=np.float32)
        rng = np.random.RandomState(42)
        emb = rng.randn(len(texts), self.FAKE_DIM).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / np.maximum(norms, 1e-12)

    def to_device(self, device: torch.device) -> None:
        pass


def _create_chunk_file(path: Path, num_chunks: int = 3) -> None:
    """Write a minimal chunk JSON file for testing."""
    chunks = []
    for i in range(num_chunks):
        chunks.append({
            "chunk_id": f"PMC123_intro_{i:03d}",
            "chunk_type": "TEXT",
            "content": f"This is test chunk number {i} about medical research.",
            "token_count": 10,
            "metadata": {
                "pmcid": "PMC123",
                "title": "Test Paper",
                "journal": "Nature",
                "year": 2024,
                "section": "Introduction",
                "subsection": "",
                "section_path": ["Introduction"],
                "chunk_type": "TEXT",
                "token_count": 10,
            },
            "metadata_header": "PMCID: PMC123\nTitle: Test Paper\nSection: Introduction",
        })

    doc = {
        "document_metadata": {
            "pmcid": "PMC123",
            "title": "Test Paper",
            "journal": "Nature",
            "year": 2024,
        },
        "chunks": chunks,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


def test_generate_embedded_document(tmp_path: Path):
    """Generator should produce a valid EmbeddedDocument."""
    chunk_file = tmp_path / "PMC123_chunks.json"
    _create_chunk_file(chunk_file, num_chunks=5)

    model = FakeEmbeddingModel()
    generator = EmbeddingGenerator(model=model, batch_size=2)

    result = generator.generate(chunk_file)

    assert isinstance(result, EmbeddedDocument)
    assert result.embedding_model == "FakeModel"
    assert result.embedding_dimension == 64
    assert len(result.chunks) == 5
    assert result.generated_at  # Should be a non-empty ISO string


def test_metadata_preservation(tmp_path: Path):
    """All chunk metadata should survive the embedding process."""
    chunk_file = tmp_path / "PMC456_chunks.json"
    _create_chunk_file(chunk_file, num_chunks=2)

    model = FakeEmbeddingModel()
    generator = EmbeddingGenerator(model=model)

    result = generator.generate(chunk_file)

    for chunk in result.chunks:
        assert chunk.chunk_id.startswith("PMC123")
        assert chunk.metadata["pmcid"] == "PMC123"
        assert chunk.metadata["title"] == "Test Paper"
        assert chunk.metadata["journal"] == "Nature"
        assert chunk.metadata["year"] == 2024
        assert chunk.metadata["section"] == "Introduction"
        assert chunk.metadata_header != ""


def test_output_schema(tmp_path: Path):
    """JSON output should match the required schema."""
    chunk_file = tmp_path / "PMC789_chunks.json"
    _create_chunk_file(chunk_file, num_chunks=2)

    model = FakeEmbeddingModel()
    generator = EmbeddingGenerator(model=model)

    result = generator.generate(chunk_file)
    doc_dict = result.to_dict()

    # Top-level keys
    assert "document_metadata" in doc_dict
    assert "embedding_model" in doc_dict
    assert "embedding_dimension" in doc_dict
    assert "generated_at" in doc_dict
    assert "chunks" in doc_dict

    # Per-chunk keys
    for chunk in doc_dict["chunks"]:
        assert "chunk_id" in chunk
        assert "embedding" in chunk
        assert "metadata" in chunk
        assert "metadata_header" in chunk
        assert isinstance(chunk["embedding"], list)
        assert len(chunk["embedding"]) == 64


def test_empty_chunks_handled(tmp_path: Path):
    """Document with no chunks should return an empty EmbeddedDocument."""
    chunk_file = tmp_path / "PMC_empty_chunks.json"
    doc = {"document_metadata": {"pmcid": "PMC_empty"}, "chunks": []}
    chunk_file.write_text(json.dumps(doc), encoding="utf-8")

    model = FakeEmbeddingModel()
    generator = EmbeddingGenerator(model=model)

    result = generator.generate(chunk_file)

    assert len(result.chunks) == 0
    assert result.embedding_model == "FakeModel"


def test_document_metadata_preserved(tmp_path: Path):
    """Document-level metadata should be passed through unchanged."""
    chunk_file = tmp_path / "PMC_meta_chunks.json"
    _create_chunk_file(chunk_file, num_chunks=1)

    model = FakeEmbeddingModel()
    generator = EmbeddingGenerator(model=model)

    result = generator.generate(chunk_file)

    assert result.document_metadata["pmcid"] == "PMC123"
    assert result.document_metadata["title"] == "Test Paper"


def test_compute_norms(tmp_path: Path):
    """Norms of embedded chunks should be approximately 1.0."""
    chunk_file = tmp_path / "PMC_norms_chunks.json"
    _create_chunk_file(chunk_file, num_chunks=5)

    model = FakeEmbeddingModel()
    generator = EmbeddingGenerator(model=model)

    result = generator.generate(chunk_file)
    norms = generator.compute_norms(result.chunks)

    assert len(norms) == 5
    for norm in norms:
        assert abs(norm - 1.0) < 1e-3
