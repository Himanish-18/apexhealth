"""
Tests for embedding pipeline (embeddings.embedding_pipeline).
Tests resume/force logic, checkpoint creation, and statistics output.
Uses a monkeypatched pipeline with a fake model — no real model loading.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from embeddings.embedding_model import EmbeddingModel
from embeddings.embedding_pipeline import EmbeddingPipeline


class FakeEmbeddingModel(EmbeddingModel):
    """Deterministic fake for pipeline tests."""

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


def _create_chunk_file(path: Path, pmcid: str = "PMC123") -> None:
    """Write a minimal chunk JSON file."""
    doc = {
        "document_metadata": {
            "pmcid": pmcid,
            "title": "Test Paper",
            "journal": "Nature",
            "year": 2024,
        },
        "chunks": [
            {
                "chunk_id": f"{pmcid}_intro_001",
                "chunk_type": "TEXT",
                "content": "This is a test chunk about medical research.",
                "token_count": 10,
                "metadata": {
                    "pmcid": pmcid,
                    "title": "Test Paper",
                    "journal": "Nature",
                    "year": 2024,
                    "section": "Introduction",
                    "subsection": "",
                    "section_path": ["Introduction"],
                    "chunk_type": "TEXT",
                    "token_count": 10,
                },
                "metadata_header": f"PMCID: {pmcid}\nTitle: Test Paper",
            },
            {
                "chunk_id": f"{pmcid}_methods_001",
                "chunk_type": "TEXT",
                "content": "Methods section content for testing.",
                "token_count": 8,
                "metadata": {
                    "pmcid": pmcid,
                    "title": "Test Paper",
                    "journal": "Nature",
                    "year": 2024,
                    "section": "Methods",
                    "subsection": "",
                    "section_path": ["Methods"],
                    "chunk_type": "TEXT",
                    "token_count": 8,
                },
                "metadata_header": f"PMCID: {pmcid}\nTitle: Test Paper",
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


def _setup_test_dirs(tmp_path: Path):
    """Create the directory structure for pipeline tests."""
    chunks_dir = tmp_path / "data" / "chunks"
    embeddings_dir = tmp_path / "data" / "embeddings"
    logs_dir = tmp_path / "data" / "logs" / "embeddings"

    chunks_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return chunks_dir, embeddings_dir, logs_dir


def _make_mock_settings(tmp_path: Path):
    """Create a mock Settings object pointing to tmp directories."""
    chunks_dir, embeddings_dir, logs_dir = _setup_test_dirs(tmp_path)

    settings = MagicMock()
    settings.chunks_dir = chunks_dir
    settings.embeddings_dir = embeddings_dir
    settings.embedding_logs_dir = logs_dir
    settings.embedding_batch_size = 32
    settings.embedding_model_name = "ncbi/MedCPT-Article-Encoder"

    return settings


def test_resume_skips_existing(tmp_path: Path):
    """Pre-existing embedding output should be skipped on resume."""
    settings = _make_mock_settings(tmp_path)

    # Create a chunk file
    _create_chunk_file(settings.chunks_dir / "PMC100_chunks.json", "PMC100")

    # Pre-create output so resume skips it
    output = settings.embeddings_dir / "PMC100_embeddings.json"
    output.write_text("{}", encoding="utf-8")

    pipeline = EmbeddingPipeline(settings)

    # Mock MedCPTEmbeddingModel to avoid loading the real model
    with patch(
        "embeddings.embedding_pipeline.MedCPTEmbeddingModel",
        return_value=FakeEmbeddingModel(),
    ):
        stats = pipeline.run(resume=True, force=False, device="cpu")

    # Should have skipped the document
    assert stats.documents_skipped == 1
    assert stats.total_documents == 0


def test_force_overwrites(tmp_path: Path):
    """Force mode should re-embed even when output exists."""
    settings = _make_mock_settings(tmp_path)

    # Create a chunk file
    _create_chunk_file(settings.chunks_dir / "PMC200_chunks.json", "PMC200")

    # Pre-create output
    output = settings.embeddings_dir / "PMC200_embeddings.json"
    output.write_text("{}", encoding="utf-8")

    pipeline = EmbeddingPipeline(settings)

    with patch(
        "embeddings.embedding_pipeline.MedCPTEmbeddingModel",
        return_value=FakeEmbeddingModel(),
    ):
        stats = pipeline.run(resume=True, force=True, device="cpu")

    # Should have processed the document
    assert stats.total_documents == 1
    assert stats.documents_skipped == 0

    # Output should be valid JSON with embeddings
    with open(output, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "chunks" in data
    assert len(data["chunks"]) == 2


def test_checkpoint_file_created(tmp_path: Path):
    """Embedding output JSON should be created for each processed document."""
    settings = _make_mock_settings(tmp_path)

    _create_chunk_file(settings.chunks_dir / "PMC300_chunks.json", "PMC300")
    _create_chunk_file(settings.chunks_dir / "PMC301_chunks.json", "PMC301")

    pipeline = EmbeddingPipeline(settings)

    with patch(
        "embeddings.embedding_pipeline.MedCPTEmbeddingModel",
        return_value=FakeEmbeddingModel(),
    ):
        stats = pipeline.run(device="cpu")

    # Both output files should exist
    assert (settings.embeddings_dir / "PMC300_embeddings.json").exists()
    assert (settings.embeddings_dir / "PMC301_embeddings.json").exists()
    assert stats.total_documents == 2


def test_statistics_saved(tmp_path: Path):
    """embedding_statistics.json should be written after the run."""
    settings = _make_mock_settings(tmp_path)

    _create_chunk_file(settings.chunks_dir / "PMC400_chunks.json", "PMC400")

    pipeline = EmbeddingPipeline(settings)

    with patch(
        "embeddings.embedding_pipeline.MedCPTEmbeddingModel",
        return_value=FakeEmbeddingModel(),
    ):
        pipeline.run(device="cpu")

    stats_file = settings.embedding_logs_dir / "embedding_statistics.json"
    assert stats_file.exists()

    with open(stats_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "total_documents" in data
    assert "total_chunks" in data
    assert "embedding_dimension" in data
    assert data["total_documents"] == 1


def test_no_chunk_files(tmp_path: Path):
    """Pipeline should handle no input files gracefully."""
    settings = _make_mock_settings(tmp_path)

    pipeline = EmbeddingPipeline(settings)

    with patch(
        "embeddings.embedding_pipeline.MedCPTEmbeddingModel",
        return_value=FakeEmbeddingModel(),
    ):
        stats = pipeline.run(device="cpu")

    assert stats.total_documents == 0
    assert stats.total_chunks == 0
