"""
Healthcare Knowledge Navigator — Semantic Chunker.

Section-aware text chunking with token-based boundaries. Processes
canonical JSON sections independently, never merging content across
section boundaries. Uses greedy sentence accumulation with configurable
overlap for continuity.

Algorithm:
    1. Walk sections recursively, tracking the section_path.
    2. For each leaf section, split content into sentences.
    3. Greedily accumulate sentences until token budget is reached.
    4. Finalize chunk, carry overlap tokens into next chunk.
    5. Handle abstract as ChunkType.ABSTRACT.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from preprocessing.chunk_metadata import build_metadata, generate_metadata_header
from preprocessing.chunker import Chunk, ChunkType, slugify_section
from preprocessing.token_counter import get_token_counter

logger = logging.getLogger(__name__)

# Sentence boundary regex — splits after sentence-ending punctuation
# followed by whitespace. Handles common abbreviations gracefully enough
# for biomedical text (e.g., "et al." won't typically end a sentence
# because the next char is usually lowercase).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class SemanticChunker:
    """Section-aware text chunker with token-based boundaries.

    Attributes:
        target_tokens: Ideal tokens per chunk.
        max_tokens: Hard ceiling per chunk.
        min_tokens: Minimum tokens — chunks below this are rejected downstream.
        overlap_tokens: Number of trailing tokens carried into the next chunk.
    """

    def __init__(
        self,
        target_tokens: int = 512,
        max_tokens: int = 600,
        min_tokens: int = 50,
        overlap_tokens: int = 75,
    ) -> None:
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self.counter = get_token_counter()

    def chunk_abstract(
        self,
        abstract_text: str,
        pmcid: str,
        doc_metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Chunk the abstract text as ABSTRACT type.

        Args:
            abstract_text: The abstract text from the canonical document.
            pmcid: The PMC identifier.
            doc_metadata: Top-level document metadata dict.

        Returns:
            List of Chunk objects for the abstract.
        """
        if not abstract_text or not abstract_text.strip():
            return []

        return self._chunk_text_block(
            text=abstract_text.strip(),
            section_path=["Abstract"],
            pmcid=pmcid,
            doc_metadata=doc_metadata,
            chunk_type=ChunkType.ABSTRACT,
        )

    def chunk_sections(
        self,
        sections: list[dict[str, Any]],
        pmcid: str,
        doc_metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Chunk all sections from the canonical document.

        Each section (and subsection) is chunked independently.
        Content never crosses section boundaries.

        Args:
            sections: List of section dicts from the canonical JSON.
            pmcid: The PMC identifier.
            doc_metadata: Top-level document metadata dict.

        Returns:
            Ordered list of Chunk objects across all sections.
        """
        chunks: list[Chunk] = []
        for section in sections:
            self._process_section(
                section=section,
                parent_path=[],
                pmcid=pmcid,
                doc_metadata=doc_metadata,
                chunks=chunks,
            )
        return chunks

    def _process_section(
        self,
        section: dict[str, Any],
        parent_path: list[str],
        pmcid: str,
        doc_metadata: dict[str, Any],
        chunks: list[Chunk],
    ) -> None:
        """Recursively process a section and its subsections.

        Args:
            section: Section dict with 'title', 'content', 'subsections'.
            parent_path: The ancestor section path.
            pmcid: The PMC identifier.
            doc_metadata: Top-level document metadata dict.
            chunks: Accumulator list to append chunks to.
        """
        title = section.get("title", "Untitled Section")
        current_path = parent_path + [title]
        content = section.get("content", "")

        # Chunk this section's own content (if any)
        if content and content.strip():
            section_chunks = self._chunk_text_block(
                text=content.strip(),
                section_path=current_path,
                pmcid=pmcid,
                doc_metadata=doc_metadata,
                chunk_type=ChunkType.TEXT,
            )
            chunks.extend(section_chunks)

        # Recursively process subsections
        subsections = section.get("subsections", [])
        for subsec in subsections:
            self._process_section(
                section=subsec,
                parent_path=current_path,
                pmcid=pmcid,
                doc_metadata=doc_metadata,
                chunks=chunks,
            )

    def _chunk_text_block(
        self,
        text: str,
        section_path: list[str],
        pmcid: str,
        doc_metadata: dict[str, Any],
        chunk_type: ChunkType,
    ) -> list[Chunk]:
        """Split a text block into token-bounded chunks.

        Uses greedy sentence accumulation with overlap.

        Args:
            text: The text to chunk.
            section_path: Hierarchical section path.
            pmcid: The PMC identifier.
            doc_metadata: Top-level document metadata dict.
            chunk_type: The type tag for produced chunks.

        Returns:
            List of Chunk objects.
        """
        sentences = _SENTENCE_SPLIT_RE.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        # Handle edge case: single sentence that fits
        if len(sentences) == 1:
            return self._finalize_single_block(
                sentences[0], section_path, pmcid, doc_metadata, chunk_type, 0
            )

        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_tokens = 0
        seq = 0

        for sentence in sentences:
            sentence_tokens = self.counter.count_tokens(sentence)

            # If a single sentence exceeds max_tokens, hard-split it
            if sentence_tokens > self.max_tokens:
                # First, finalize any accumulated sentences
                if current_sentences:
                    chunk = self._build_chunk(
                        " ".join(current_sentences),
                        section_path, pmcid, doc_metadata, chunk_type, seq,
                    )
                    chunks.append(chunk)
                    seq += 1
                    current_sentences = []
                    current_tokens = 0

                # Hard-split the oversized sentence
                hard_chunks = self._hard_split(
                    sentence, section_path, pmcid, doc_metadata, chunk_type, seq
                )
                chunks.extend(hard_chunks)
                seq += len(hard_chunks)
                continue

            # Check if adding this sentence would exceed target
            projected = current_tokens + sentence_tokens
            if projected > self.target_tokens and current_sentences:
                # Finalize current chunk
                chunk = self._build_chunk(
                    " ".join(current_sentences),
                    section_path, pmcid, doc_metadata, chunk_type, seq,
                )
                chunks.append(chunk)
                seq += 1

                # Build overlap from trailing sentences
                overlap_sentences = self._compute_overlap(current_sentences)
                current_sentences = overlap_sentences + [sentence]
                current_tokens = self.counter.count_tokens(
                    " ".join(current_sentences)
                )
            else:
                current_sentences.append(sentence)
                current_tokens = projected

        # Finalize remaining sentences
        if current_sentences:
            chunk = self._build_chunk(
                " ".join(current_sentences),
                section_path, pmcid, doc_metadata, chunk_type, seq,
            )
            chunks.append(chunk)

        return chunks

    def _build_chunk(
        self,
        content: str,
        section_path: list[str],
        pmcid: str,
        doc_metadata: dict[str, Any],
        chunk_type: ChunkType,
        seq: int,
    ) -> Chunk:
        """Build a single Chunk with metadata and header.

        Args:
            content: The chunk text content.
            section_path: Hierarchical section path.
            pmcid: The PMC identifier.
            doc_metadata: Top-level document metadata dict.
            chunk_type: Chunk type enum value.
            seq: Sequence number within the section.

        Returns:
            A fully populated Chunk instance.
        """
        section_slug = slugify_section(section_path[-1]) if section_path else "root"
        chunk_id = f"{pmcid}_{section_slug}_{seq:03d}"
        token_count = self.counter.count_tokens(content)

        metadata = build_metadata(
            pmcid=pmcid,
            doc_metadata=doc_metadata,
            section_path=section_path,
            chunk_type=chunk_type.value,
            token_count=token_count,
        )
        header = generate_metadata_header(metadata)

        return Chunk(
            chunk_id=chunk_id,
            chunk_type=chunk_type.value,
            content=content,
            token_count=token_count,
            metadata=metadata,
            metadata_header=header,
        )

    def _finalize_single_block(
        self,
        text: str,
        section_path: list[str],
        pmcid: str,
        doc_metadata: dict[str, Any],
        chunk_type: ChunkType,
        seq: int,
    ) -> list[Chunk]:
        """Handle the case where there is only a single text block.

        If the single block exceeds max_tokens, hard-split it.
        Otherwise, produce a single chunk.

        Args:
            text: The text block.
            section_path: Hierarchical section path.
            pmcid: The PMC identifier.
            doc_metadata: Top-level document metadata dict.
            chunk_type: Chunk type enum value.
            seq: Starting sequence number.

        Returns:
            List of Chunk objects (usually one, more if hard-split).
        """
        token_count = self.counter.count_tokens(text)
        if token_count > self.max_tokens:
            return self._hard_split(
                text, section_path, pmcid, doc_metadata, chunk_type, seq
            )
        return [
            self._build_chunk(
                text, section_path, pmcid, doc_metadata, chunk_type, seq
            )
        ]

    def _hard_split(
        self,
        text: str,
        section_path: list[str],
        pmcid: str,
        doc_metadata: dict[str, Any],
        chunk_type: ChunkType,
        start_seq: int,
    ) -> list[Chunk]:
        """Hard-split an oversized text on token boundaries.

        This is the fallback for extremely long sentences or text blocks
        that cannot be split at sentence boundaries.

        Args:
            text: The oversized text.
            section_path: Hierarchical section path.
            pmcid: The PMC identifier.
            doc_metadata: Top-level document metadata dict.
            chunk_type: Chunk type enum value.
            start_seq: Starting sequence number.

        Returns:
            List of Chunk objects from the hard split.
        """
        chunks: list[Chunk] = []
        remaining = text
        seq = start_seq

        while remaining:
            truncated = self.counter.truncate_to_tokens(remaining, self.target_tokens)
            if not truncated:
                break

            chunk = self._build_chunk(
                truncated, section_path, pmcid, doc_metadata, chunk_type, seq
            )
            chunks.append(chunk)
            seq += 1

            # Advance past the truncated portion
            remaining = remaining[len(truncated):].strip()

        return chunks

    def _compute_overlap(self, sentences: list[str]) -> list[str]:
        """Compute the overlap sentences from the end of a chunk.

        Walks backwards through sentences until the overlap token
        budget is met.

        Args:
            sentences: The sentences from the finalized chunk.

        Returns:
            List of trailing sentences that form the overlap.
        """
        if not sentences or self.overlap_tokens <= 0:
            return []

        overlap: list[str] = []
        token_count = 0

        for sent in reversed(sentences):
            sent_tokens = self.counter.count_tokens(sent)
            if token_count + sent_tokens > self.overlap_tokens:
                break
            overlap.insert(0, sent)
            token_count += sent_tokens

        return overlap
