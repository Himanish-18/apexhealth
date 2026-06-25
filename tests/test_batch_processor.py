"""
Tests for batch processor (embeddings.batch_processor).
Tests batch splitting, iteration, memory cleanup, and edge cases.
"""

import pytest

from embeddings.batch_processor import (
    create_batches,
    estimate_batch_memory_mb,
    iter_batches,
)


def test_batch_splitting_exact():
    """100 items with batch_size=25 should produce 4 batches."""
    items = list(range(100))
    batches = create_batches(items, batch_size=25)

    assert len(batches) == 4
    assert all(len(b) == 25 for b in batches)


def test_batch_splitting_remainder():
    """100 items with batch_size=32 should produce 4 batches (last smaller)."""
    items = list(range(100))
    batches = create_batches(items, batch_size=32)

    assert len(batches) == 4
    assert len(batches[0]) == 32
    assert len(batches[1]) == 32
    assert len(batches[2]) == 32
    assert len(batches[3]) == 4  # remainder


def test_empty_input():
    """Empty list should return empty list of batches."""
    batches = create_batches([], batch_size=32)
    assert batches == []


def test_single_item():
    """Single item should produce one batch with one item."""
    batches = create_batches(["a"], batch_size=32)
    assert len(batches) == 1
    assert batches[0] == ["a"]


def test_batch_size_larger_than_input():
    """When batch_size > len(items), should produce a single batch."""
    items = list(range(5))
    batches = create_batches(items, batch_size=100)

    assert len(batches) == 1
    assert len(batches[0]) == 5


def test_invalid_batch_size():
    """batch_size <= 0 should raise ValueError."""
    with pytest.raises(ValueError, match="batch_size must be positive"):
        create_batches([1, 2, 3], batch_size=0)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        create_batches([1, 2, 3], batch_size=-5)


def test_iter_batches_matches_create_batches():
    """iter_batches should yield the same results as create_batches."""
    items = list(range(73))
    created = create_batches(items, batch_size=20)
    iterated = list(iter_batches(items, batch_size=20))

    assert created == iterated


def test_iter_batches_empty():
    """iter_batches on empty input should yield nothing."""
    result = list(iter_batches([], batch_size=10))
    assert result == []


def test_iter_batches_invalid_batch_size():
    """iter_batches with invalid batch_size should raise ValueError."""
    with pytest.raises(ValueError):
        list(iter_batches([1], batch_size=0))


def test_estimate_batch_memory_positive():
    """Memory estimate should return a positive value."""
    mem = estimate_batch_memory_mb(batch_size=32, embedding_dim=768)
    assert mem > 0


def test_estimate_batch_memory_scales_with_batch_size():
    """Larger batch size should require more memory."""
    mem_small = estimate_batch_memory_mb(batch_size=16, embedding_dim=768)
    mem_large = estimate_batch_memory_mb(batch_size=64, embedding_dim=768)
    assert mem_large > mem_small


def test_all_items_present_in_batches():
    """Every item should appear exactly once across all batches."""
    items = list(range(50))
    batches = create_batches(items, batch_size=7)

    flattened = [item for batch in batches for item in batch]
    assert flattened == items
