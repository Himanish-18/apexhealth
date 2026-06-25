"""
Healthcare Knowledge Navigator — Batch Processor.

Producer-consumer architecture for embedding generation using threads.
Threads (not processes) are used because the GPU model lives on a single
device and cannot be pickled — threads share memory, and the GIL is
released during torch GPU operations.

Architecture::

    Thread 1 (Producer):  Read chunk files → enqueue file paths
    Thread 2 (GPU Worker): Dequeue → embed → enqueue results
    Thread 3 (Consumer):  Dequeue results → validate → write to disk

This module provides batch-splitting utilities used by the pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

import torch

logger = logging.getLogger(__name__)


def create_batches(
    items: list[Any],
    batch_size: int,
) -> list[list[Any]]:
    """Split a list of items into fixed-size batches.

    Args:
        items: The list to split.
        batch_size: Maximum number of items per batch.

    Returns:
        List of batches, where each batch is a list of items.
        The last batch may contain fewer items.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    if not items:
        return []

    return [
        items[i: i + batch_size]
        for i in range(0, len(items), batch_size)
    ]


def iter_batches(
    items: list[Any],
    batch_size: int,
) -> Iterator[list[Any]]:
    """Iterate over items in fixed-size batches (memory-efficient).

    Unlike ``create_batches``, this yields one batch at a time
    without materializing the full list of batches.

    Args:
        items: The list to iterate over.
        batch_size: Maximum number of items per batch.

    Yields:
        Batches of items as lists.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    for i in range(0, len(items), batch_size):
        yield items[i: i + batch_size]


def cleanup_gpu_memory() -> None:
    """Free unused GPU memory after a batch.

    Calls ``torch.cuda.empty_cache()`` if CUDA is available.
    This is a no-op on CPU and MPS.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU memory cache cleared")


def estimate_batch_memory_mb(
    batch_size: int,
    embedding_dim: int,
    avg_token_length: int = 512,
) -> float:
    """Estimate the approximate memory needed for a batch.

    This is a rough estimate useful for logging and diagnostics.
    The actual memory usage depends on model architecture, attention
    patterns, and padding.

    Args:
        batch_size: Number of items in the batch.
        embedding_dim: Dimensionality of the output embedding.
        avg_token_length: Average number of tokens per input text.

    Returns:
        Estimated memory usage in megabytes.
    """
    # Input tokens (int64): batch_size × avg_token_length × 8 bytes
    input_bytes = batch_size * avg_token_length * 8

    # Hidden states (float32): batch_size × avg_token_length × embedding_dim × 4 bytes
    hidden_bytes = batch_size * avg_token_length * embedding_dim * 4

    # Output embeddings (float32): batch_size × embedding_dim × 4 bytes
    output_bytes = batch_size * embedding_dim * 4

    total_bytes = input_bytes + hidden_bytes + output_bytes
    return total_bytes / (1024 * 1024)
