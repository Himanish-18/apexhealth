"""
Healthcare Knowledge Navigator — Table Chunker.

Converts each table from the canonical JSON into its own chunk.
Tables are never split — splitting destroys tabular semantics.
Figure captions are also handled here as separate FIGURE_CAPTION chunks.
"""

from __future__ import annotations

import logging
from typing import Any

from preprocessing.chunk_metadata import build_metadata, generate_metadata_header
from preprocessing.chunker import Chunk, ChunkType, slugify_section
from preprocessing.token_counter import get_token_counter

logger = logging.getLogger(__name__)


class TableChunker:
    """Converts tables and figure captions into standalone chunks.

    Tables are kept whole regardless of token count — splitting a table
    destroys its structural meaning. A warning is logged if a table
    exceeds the max token threshold.

    Attributes:
        max_tokens: Soft warning threshold for oversized tables.
    """

    def __init__(self, max_tokens: int = 600) -> None:
        self.max_tokens = max_tokens
        self.counter = get_token_counter()

    def chunk_tables(
        self,
        tables: list[dict[str, Any]],
        pmcid: str,
        doc_metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Convert each table into a single TABLE chunk.

        Args:
            tables: List of table dicts from the canonical JSON.
            pmcid: The PMC identifier.
            doc_metadata: Top-level document metadata dict.

        Returns:
            List of TABLE chunks, one per table.
        """
        chunks: list[Chunk] = []

        for idx, table in enumerate(tables):
            table_id = table.get("id", f"table_{idx}")
            caption = table.get("caption", "")
            markdown = table.get("markdown", "")
            section = table.get("section", "")

            if not markdown:
                continue

            # Build content: caption + markdown
            content_parts = []
            if caption:
                content_parts.append(caption)
            content_parts.append(markdown)
            content = "\n\n".join(content_parts)

            token_count = self.counter.count_tokens(content)

            if token_count > self.max_tokens:
                logger.warning(
                    "Table %s in %s exceeds max tokens (%d > %d). "
                    "Keeping whole to preserve tabular semantics.",
                    table_id, pmcid, token_count, self.max_tokens,
                )

            section_path = [section] if section else ["Tables"]
            section_slug = slugify_section(table_id)
            chunk_id = f"{pmcid}_{section_slug}_{idx:03d}"

            metadata = build_metadata(
                pmcid=pmcid,
                doc_metadata=doc_metadata,
                section_path=section_path,
                chunk_type=ChunkType.TABLE.value,
                token_count=token_count,
            )
            header = generate_metadata_header(metadata)

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    chunk_type=ChunkType.TABLE.value,
                    content=content,
                    token_count=token_count,
                    metadata=metadata,
                    metadata_header=header,
                )
            )

        return chunks

    def chunk_figures(
        self,
        figures: list[dict[str, Any]],
        pmcid: str,
        doc_metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Convert each figure caption into a FIGURE_CAPTION chunk.

        Only the caption text is included — no image data.

        Args:
            figures: List of figure dicts from the canonical JSON.
            pmcid: The PMC identifier.
            doc_metadata: Top-level document metadata dict.

        Returns:
            List of FIGURE_CAPTION chunks.
        """
        chunks: list[Chunk] = []

        for idx, figure in enumerate(figures):
            fig_id = figure.get("id", f"fig_{idx}")
            caption = figure.get("caption", "")

            if not caption or not caption.strip():
                continue

            content = caption.strip()
            token_count = self.counter.count_tokens(content)

            section_path = ["Figures"]
            section_slug = slugify_section(fig_id)
            chunk_id = f"{pmcid}_{section_slug}_{idx:03d}"

            metadata = build_metadata(
                pmcid=pmcid,
                doc_metadata=doc_metadata,
                section_path=section_path,
                chunk_type=ChunkType.FIGURE_CAPTION.value,
                token_count=token_count,
            )
            header = generate_metadata_header(metadata)

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    chunk_type=ChunkType.FIGURE_CAPTION.value,
                    content=content,
                    token_count=token_count,
                    metadata=metadata,
                    metadata_header=header,
                )
            )

        return chunks
