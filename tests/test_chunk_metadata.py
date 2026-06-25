"""
Tests for chunk metadata builder (preprocessing.chunk_metadata).
"""

from preprocessing.chunk_metadata import build_metadata, generate_metadata_header
from preprocessing.chunker import ChunkType


def _sample_doc_metadata():
    return {
        "pmcid": "PMC123456",
        "title": "Efficacy of Drug X",
        "journal": "Nature Medicine",
        "year": 2024,
    }


def test_build_metadata_populates_all_fields():
    """Metadata should have all required fields populated."""
    meta = build_metadata(
        pmcid="PMC123456",
        doc_metadata=_sample_doc_metadata(),
        section_path=["Methods", "Statistical Analysis"],
        chunk_type=ChunkType.TEXT.value,
        token_count=450,
    )
    assert meta.pmcid == "PMC123456"
    assert meta.title == "Efficacy of Drug X"
    assert meta.journal == "Nature Medicine"
    assert meta.year == 2024
    assert meta.section == "Methods"
    assert meta.subsection == "Statistical Analysis"
    assert meta.section_path == ["Methods", "Statistical Analysis"]
    assert meta.chunk_type == "TEXT"
    assert meta.token_count == 450


def test_build_metadata_single_section():
    """Single-level section should have empty subsection."""
    meta = build_metadata(
        pmcid="PMC123456",
        doc_metadata=_sample_doc_metadata(),
        section_path=["Introduction"],
        chunk_type=ChunkType.TEXT.value,
        token_count=300,
    )
    assert meta.section == "Introduction"
    assert meta.subsection == ""


def test_generate_metadata_header_format():
    """Header should contain PMCID, journal, year, section, and subsection."""
    meta = build_metadata(
        pmcid="PMC123456",
        doc_metadata=_sample_doc_metadata(),
        section_path=["Results", "Efficacy Analysis"],
        chunk_type=ChunkType.TEXT.value,
        token_count=500,
    )
    header = generate_metadata_header(meta)

    assert "PMCID: PMC123456" in header
    assert "Journal: Nature Medicine" in header
    assert "Year: 2024" in header
    assert "Section: Results" in header
    assert "Subsection: Efficacy Analysis" in header


def test_generate_metadata_header_no_subsection():
    """Header without subsection should omit that line."""
    meta = build_metadata(
        pmcid="PMC999",
        doc_metadata={"title": "Test", "journal": "Lancet", "year": 2023},
        section_path=["Abstract"],
        chunk_type=ChunkType.ABSTRACT.value,
        token_count=200,
    )
    header = generate_metadata_header(meta)
    assert "Subsection:" not in header


def test_section_path_is_independent_copy():
    """section_path should be an independent copy, not a reference."""
    original_path = ["Methods", "Design"]
    meta = build_metadata(
        pmcid="PMC1",
        doc_metadata=_sample_doc_metadata(),
        section_path=original_path,
        chunk_type=ChunkType.TEXT.value,
        token_count=100,
    )
    original_path.append("Mutated")
    assert len(meta.section_path) == 2  # Should not be affected
