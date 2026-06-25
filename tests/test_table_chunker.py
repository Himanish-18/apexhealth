"""
Tests for the table chunker (preprocessing.table_chunker).
"""

from preprocessing.table_chunker import TableChunker
from preprocessing.chunker import ChunkType


def _sample_metadata():
    return {
        "pmcid": "PMC123456",
        "title": "Test Paper",
        "journal": "Lancet",
        "year": 2023,
    }


def test_one_table_one_chunk():
    """Each table should produce exactly one chunk."""
    chunker = TableChunker(max_tokens=600)
    tables = [
        {
            "id": "T1",
            "caption": "Baseline characteristics",
            "markdown": "| Age | Count |\n|---|---|\n| 40 | 120 |",
            "section": "Results",
        },
    ]
    chunks = chunker.chunk_tables(tables, "PMC123456", _sample_metadata())

    assert len(chunks) == 1
    assert chunks[0].chunk_type == ChunkType.TABLE.value


def test_table_content_includes_caption_and_markdown():
    """Table chunk content should include both caption and markdown."""
    chunker = TableChunker()
    tables = [
        {
            "id": "T2",
            "caption": "Drug dosages",
            "markdown": "| Drug | Dose |\n|---|---|\n| X | 500mg |",
            "section": "Methods",
        },
    ]
    chunks = chunker.chunk_tables(tables, "PMC123", _sample_metadata())

    assert "Drug dosages" in chunks[0].content
    assert "| Drug | Dose |" in chunks[0].content


def test_table_id_in_chunk_id():
    """Table chunk ID should reference the table ID."""
    chunker = TableChunker()
    tables = [
        {
            "id": "T3",
            "caption": "Results",
            "markdown": "| A | B |\n|---|---|\n| 1 | 2 |",
            "section": "Results",
        },
    ]
    chunks = chunker.chunk_tables(tables, "PMC555", _sample_metadata())

    assert "t3" in chunks[0].chunk_id.lower()


def test_empty_markdown_skipped():
    """Tables with no markdown content should be skipped."""
    chunker = TableChunker()
    tables = [
        {
            "id": "T4",
            "caption": "Empty table",
            "markdown": "",
            "section": "Results",
        },
    ]
    chunks = chunker.chunk_tables(tables, "PMC1", _sample_metadata())
    assert len(chunks) == 0


def test_figure_caption_chunk():
    """Figure captions should produce FIGURE_CAPTION chunks."""
    chunker = TableChunker()
    figures = [
        {
            "id": "fig_1",
            "caption": "Kaplan-Meier survival curve for treatment vs control groups.",
        },
    ]
    chunks = chunker.chunk_figures(figures, "PMC123", _sample_metadata())

    assert len(chunks) == 1
    assert chunks[0].chunk_type == ChunkType.FIGURE_CAPTION.value
    assert "Kaplan-Meier" in chunks[0].content


def test_figure_empty_caption_skipped():
    """Figures with empty captions should be skipped."""
    chunker = TableChunker()
    figures = [
        {"id": "fig_2", "caption": ""},
    ]
    chunks = chunker.chunk_figures(figures, "PMC1", _sample_metadata())
    assert len(chunks) == 0


def test_multiple_tables_produce_multiple_chunks():
    """Multiple tables should each produce one chunk."""
    chunker = TableChunker()
    tables = [
        {"id": "T1", "caption": "Table 1", "markdown": "| A |\n|---|\n| 1 |", "section": "Results"},
        {"id": "T2", "caption": "Table 2", "markdown": "| B |\n|---|\n| 2 |", "section": "Methods"},
    ]
    chunks = chunker.chunk_tables(tables, "PMC1", _sample_metadata())
    assert len(chunks) == 2


def test_table_section_in_metadata():
    """Table chunks should have their parent section in metadata."""
    chunker = TableChunker()
    tables = [
        {
            "id": "T5",
            "caption": "Outcomes",
            "markdown": "| Col |\n|---|\n| val |",
            "section": "Results",
        },
    ]
    chunks = chunker.chunk_tables(tables, "PMC1", _sample_metadata())
    assert chunks[0].metadata.section == "Results"
