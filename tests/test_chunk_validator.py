"""
Tests for the chunk validator (preprocessing.chunk_validator).
"""

from preprocessing.chunk_validator import ChunkValidator
from preprocessing.chunker import Chunk, ChunkMetadata, ChunkType


def _make_chunk(
    content: str = "Valid test content with enough tokens to pass validation easily.",
    token_count: int = 100,
    pmcid: str = "PMC123",
    section: str = "Introduction",
    chunk_type: str = "TEXT",
    chunk_id: str = "PMC123_intro_001",
) -> Chunk:
    """Factory for creating test chunks."""
    return Chunk(
        chunk_id=chunk_id,
        chunk_type=chunk_type,
        content=content,
        token_count=token_count,
        metadata=ChunkMetadata(
            pmcid=pmcid,
            title="Test Paper",
            journal="Test Journal",
            year=2024,
            section=section,
            subsection="",
            section_path=[section],
            chunk_type=chunk_type,
            token_count=token_count,
        ),
        metadata_header="PMCID: PMC123",
    )


def test_valid_chunk_passes():
    """A chunk with valid content and metadata should pass validation."""
    validator = ChunkValidator(min_tokens=50)
    chunks = [_make_chunk()]
    valid, rejected, report = validator.validate(chunks)

    assert len(valid) == 1
    assert len(rejected) == 0
    assert report.total_valid == 1
    assert report.total_rejected == 0


def test_below_min_tokens_rejected():
    """A chunk below the minimum token count should be rejected."""
    validator = ChunkValidator(min_tokens=50)
    chunk = _make_chunk(token_count=30)
    valid, rejected, report = validator.validate([chunk])

    assert len(valid) == 0
    assert len(rejected) == 1
    assert "below_min_tokens" in report.rejections[0]["reasons"][0]


def test_empty_content_rejected():
    """A chunk with empty content should be rejected."""
    validator = ChunkValidator(min_tokens=50)
    chunk = _make_chunk(content="", token_count=0)
    valid, rejected, report = validator.validate([chunk])

    assert len(valid) == 0
    assert len(rejected) == 1
    assert any("empty_content" in r for r in report.rejections[0]["reasons"])


def test_whitespace_only_content_rejected():
    """A chunk with whitespace-only content should be rejected."""
    validator = ChunkValidator(min_tokens=50)
    chunk = _make_chunk(content="   \n\t  ", token_count=0)
    valid, rejected, report = validator.validate([chunk])

    assert len(rejected) == 1


def test_missing_pmcid_rejected():
    """A chunk with missing PMCID should be rejected."""
    validator = ChunkValidator(min_tokens=50)
    chunk = _make_chunk(pmcid="")
    valid, rejected, report = validator.validate([chunk])

    assert len(rejected) == 1
    assert any("missing_pmcid" in r for r in report.rejections[0]["reasons"])


def test_mixed_valid_and_invalid():
    """Validator should correctly separate valid and invalid chunks."""
    validator = ChunkValidator(min_tokens=50)
    chunks = [
        _make_chunk(chunk_id="valid_001"),
        _make_chunk(content="", token_count=0, chunk_id="invalid_001"),
        _make_chunk(chunk_id="valid_002"),
    ]
    valid, rejected, report = validator.validate(chunks)

    assert len(valid) == 2
    assert len(rejected) == 1
    assert report.total_input == 3


def test_validation_report_structure():
    """Report should have correct counts and rejection details."""
    validator = ChunkValidator(min_tokens=50)
    chunks = [
        _make_chunk(token_count=10, chunk_id="small_001"),
    ]
    valid, rejected, report = validator.validate(chunks)

    assert report.total_input == 1
    assert report.total_valid == 0
    assert report.total_rejected == 1
    assert len(report.rejections) == 1
    assert report.rejections[0]["chunk_id"] == "small_001"
