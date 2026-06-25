"""
Healthcare Knowledge Navigator — Chunk Metadata Builder.

Constructs ChunkMetadata instances and generates retrieval header
strings for each chunk. The metadata header is stored separately
from content and is NOT prepended — downstream consumers decide
whether to inject it at retrieval time.
"""

from __future__ import annotations

from typing import Any

from preprocessing.chunker import ChunkMetadata


def build_metadata(
    pmcid: str,
    doc_metadata: dict[str, Any],
    section_path: list[str],
    chunk_type: str,
    token_count: int,
) -> ChunkMetadata:
    """Build a ChunkMetadata instance from document-level metadata.

    Args:
        pmcid: The PMC identifier.
        doc_metadata: Top-level document metadata dict from the canonical JSON.
        section_path: Hierarchical path (e.g., ["Methods", "Statistical Analysis"]).
        chunk_type: ChunkType value string.
        token_count: Number of tokens in the chunk content.

    Returns:
        A populated ChunkMetadata instance.
    """
    section = section_path[0] if section_path else ""
    subsection = section_path[-1] if len(section_path) > 1 else ""

    return ChunkMetadata(
        pmcid=pmcid,
        title=doc_metadata.get("title", ""),
        journal=doc_metadata.get("journal", ""),
        year=doc_metadata.get("year"),
        section=section,
        subsection=subsection,
        section_path=list(section_path),
        chunk_type=chunk_type,
        token_count=token_count,
    )


def generate_metadata_header(metadata: ChunkMetadata) -> str:
    """Generate a human-readable retrieval header from chunk metadata.

    The header is stored in the chunk but NOT prepended to content.
    Downstream consumers (e.g., embedding or retrieval modules) decide
    whether to inject it.

    Args:
        metadata: The ChunkMetadata instance.

    Returns:
        A multi-line string header for retrieval context.

    Example::

        PMCID: PMC123456
        Journal: Nature Medicine
        Year: 2024
        Section: Results
        Subsection: Efficacy Analysis
    """
    lines = [
        f"PMCID: {metadata.pmcid}",
    ]

    if metadata.title:
        lines.append(f"Title: {metadata.title}")
    if metadata.journal:
        lines.append(f"Journal: {metadata.journal}")
    if metadata.year is not None:
        lines.append(f"Year: {metadata.year}")
    if metadata.section:
        lines.append(f"Section: {metadata.section}")
    if metadata.subsection:
        lines.append(f"Subsection: {metadata.subsection}")

    return "\n".join(lines)
