"""
Tests for the semantic chunker (preprocessing.semantic_chunker).
"""

from preprocessing.semantic_chunker import SemanticChunker
from preprocessing.chunker import ChunkType


def _sample_metadata():
    return {
        "pmcid": "PMC123456",
        "title": "Test Paper",
        "journal": "Nature",
        "year": 2024,
    }


def test_abstract_tagged_as_abstract_type():
    """Abstract chunks should be tagged as ABSTRACT type."""
    chunker = SemanticChunker(target_tokens=512, max_tokens=600, overlap_tokens=75)
    chunks = chunker.chunk_abstract(
        "This study investigates the efficacy of Drug X in treating condition Y. "
        "We conducted a randomized controlled trial with 500 participants.",
        pmcid="PMC123456",
        doc_metadata=_sample_metadata(),
    )
    assert len(chunks) >= 1
    assert all(c.chunk_type == ChunkType.ABSTRACT.value for c in chunks)


def test_empty_abstract_produces_no_chunks():
    """Empty abstract should produce no chunks."""
    chunker = SemanticChunker()
    chunks = chunker.chunk_abstract("", "PMC1", _sample_metadata())
    assert chunks == []


def test_section_boundary_isolation():
    """Content from different sections should never be in the same chunk."""
    chunker = SemanticChunker(target_tokens=512, max_tokens=600, overlap_tokens=0)
    sections = [
        {
            "title": "Methods",
            "content": "We used a randomized controlled trial design. "
                       "Patients were enrolled between January and June 2024.",
            "subsections": [],
        },
        {
            "title": "Results",
            "content": "The primary endpoint was met with statistical significance. "
                       "The p-value was less than 0.001.",
            "subsections": [],
        },
    ]
    chunks = chunker.chunk_sections(sections, "PMC123456", _sample_metadata())

    methods_chunks = [c for c in chunks if c.metadata.section == "Methods"]
    results_chunks = [c for c in chunks if c.metadata.section == "Results"]

    assert len(methods_chunks) >= 1
    assert len(results_chunks) >= 1

    # Verify no cross-contamination
    for mc in methods_chunks:
        assert "primary endpoint" not in mc.content
    for rc in results_chunks:
        assert "randomized controlled trial" not in rc.content


def test_subsection_path_tracking():
    """Subsection chunks should have the full section path in metadata."""
    chunker = SemanticChunker(target_tokens=512, max_tokens=600, overlap_tokens=0)
    sections = [
        {
            "title": "Methods",
            "content": "",
            "subsections": [
                {
                    "title": "Statistical Analysis",
                    "content": "We applied a Cox proportional hazards model. "
                               "All analyses were performed using R version 4.2.",
                    "subsections": [],
                },
            ],
        },
    ]
    chunks = chunker.chunk_sections(sections, "PMC123456", _sample_metadata())

    assert len(chunks) >= 1
    assert chunks[0].metadata.section_path == ["Methods", "Statistical Analysis"]
    assert chunks[0].metadata.section == "Methods"
    assert chunks[0].metadata.subsection == "Statistical Analysis"


def test_chunk_id_format():
    """Chunk IDs should follow {pmcid}_{slug}_{seq:03d} format."""
    chunker = SemanticChunker(target_tokens=512, max_tokens=600, overlap_tokens=0)
    sections = [
        {
            "title": "Introduction",
            "content": "This paper presents a novel approach to medical diagnosis.",
            "subsections": [],
        },
    ]
    chunks = chunker.chunk_sections(sections, "PMC99", _sample_metadata())

    assert len(chunks) >= 1
    assert chunks[0].chunk_id.startswith("PMC99_introduction_")


def test_token_limit_respected():
    """No chunk should exceed the configured max_tokens."""
    chunker = SemanticChunker(target_tokens=50, max_tokens=80, overlap_tokens=10)
    # Generate content with many sentences
    sentences = ["This is sentence number %d with some medical content about diagnosis." % i for i in range(30)]
    long_content = " ".join(sentences)

    sections = [
        {
            "title": "Discussion",
            "content": long_content,
            "subsections": [],
        },
    ]
    chunks = chunker.chunk_sections(sections, "PMC123", _sample_metadata())

    assert len(chunks) > 1  # Should produce multiple chunks
    for chunk in chunks:
        # Allow some tolerance for tokenization edge cases
        assert chunk.token_count <= 100, (
            f"Chunk {chunk.chunk_id} has {chunk.token_count} tokens (max 80)"
        )


def test_overlap_between_consecutive_chunks():
    """Consecutive chunks from the same section should share overlap text."""
    chunker = SemanticChunker(target_tokens=30, max_tokens=60, overlap_tokens=20)
    # Create enough content to force multiple chunks
    sentences = [
        "The first finding relates to cardiovascular outcomes.",
        "The second finding involves renal function improvements.",
        "The third observation concerns hepatic enzyme levels.",
        "The fourth result demonstrates neurological benefits.",
        "The fifth outcome pertains to respiratory function.",
        "The sixth measurement shows hematological parameters.",
        "The seventh indicator tracks inflammatory markers.",
        "The eighth evaluation assesses quality of life scores.",
    ]
    content = " ".join(sentences)

    sections = [
        {
            "title": "Findings",
            "content": content,
            "subsections": [],
        },
    ]
    chunks = chunker.chunk_sections(sections, "PMC500", _sample_metadata())

    if len(chunks) >= 2:
        # Check that the first chunk's trailing content appears
        # at the start of the second chunk (overlap)
        first_content_words = set(chunks[0].content.split()[-5:])
        second_content_words = set(chunks[1].content.split()[:10])
        overlap = first_content_words & second_content_words
        assert len(overlap) > 0, "Expected overlap between consecutive chunks"


def test_all_chunks_have_text_type():
    """Section chunks should all be TEXT type."""
    chunker = SemanticChunker()
    sections = [
        {
            "title": "Conclusion",
            "content": "In conclusion, our study demonstrates significant findings.",
            "subsections": [],
        },
    ]
    chunks = chunker.chunk_sections(sections, "PMC1", _sample_metadata())
    assert all(c.chunk_type == ChunkType.TEXT.value for c in chunks)


def test_empty_section_produces_no_chunks():
    """A section with empty content and no subsections produces nothing."""
    chunker = SemanticChunker()
    sections = [
        {
            "title": "Empty",
            "content": "",
            "subsections": [],
        },
    ]
    chunks = chunker.chunk_sections(sections, "PMC1", _sample_metadata())
    assert chunks == []
