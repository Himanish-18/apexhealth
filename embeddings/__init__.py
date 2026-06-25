"""
Healthcare Knowledge Navigator — Embeddings Module.

Handles loading embedding models and generating dense vector
representations of text chunks for semantic retrieval.

Components:
    - Abstract EmbeddingModel interface (for model swapping)
    - MedCPT Article Encoder implementation
    - Batch embedding utilities
    - Embedding validation and statistics
    - Device management (CUDA/MPS/CPU)
    - Pipeline orchestrator with checkpointing

Imports are lazy to avoid requiring heavy dependencies (torch,
sentence-transformers) when only lightweight components are needed.
"""

from embeddings.batch_processor import (
    cleanup_gpu_memory,
    create_batches,
    estimate_batch_memory_mb,
    iter_batches,
)
from embeddings.device_manager import DeviceInfo, detect_device
from embeddings.embedding_statistics import EmbeddingStats, StatisticsCollector
from embeddings.embedding_validator import (
    EmbeddingValidationReport,
    EmbeddingValidator,
)

__all__ = [
    # Device management
    "DeviceInfo",
    "detect_device",
    # Model interface (import from submodule directly)
    "EmbeddingModel",
    "MedCPTEmbeddingModel",
    # Generator (import from submodule directly)
    "EmbeddingGenerator",
    "EmbeddedChunk",
    "EmbeddedDocument",
    # Batch processing
    "create_batches",
    "iter_batches",
    "cleanup_gpu_memory",
    "estimate_batch_memory_mb",
    # Validation
    "EmbeddingValidator",
    "EmbeddingValidationReport",
    # Statistics
    "StatisticsCollector",
    "EmbeddingStats",
    # Pipeline (import from submodule directly)
    "EmbeddingPipeline",
]


def __getattr__(name: str):
    """Lazy imports for heavy dependencies (torch, sentence-transformers)."""
    if name in ("EmbeddingModel", "MedCPTEmbeddingModel"):
        from embeddings.embedding_model import EmbeddingModel, MedCPTEmbeddingModel
        return {"EmbeddingModel": EmbeddingModel, "MedCPTEmbeddingModel": MedCPTEmbeddingModel}[name]

    if name in ("EmbeddingGenerator", "EmbeddedChunk", "EmbeddedDocument"):
        from embeddings.embedding_generator import (
            EmbeddedChunk,
            EmbeddedDocument,
            EmbeddingGenerator,
        )
        return {"EmbeddingGenerator": EmbeddingGenerator, "EmbeddedChunk": EmbeddedChunk, "EmbeddedDocument": EmbeddedDocument}[name]

    if name == "EmbeddingPipeline":
        from embeddings.embedding_pipeline import EmbeddingPipeline
        return EmbeddingPipeline

    raise AttributeError(f"module 'embeddings' has no attribute {name!r}")
