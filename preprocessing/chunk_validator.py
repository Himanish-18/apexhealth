"""
Healthcare Knowledge Navigator — Chunk Validator.

Validates and filters chunks before final output. Ensures every
chunk meets minimum quality standards for downstream embedding
and retrieval.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from preprocessing.chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Report summarizing chunk validation results.

    Attributes:
        total_input: Number of chunks received for validation.
        total_valid: Number of chunks that passed validation.
        total_rejected: Number of chunks that were rejected.
        rejections: List of dicts describing each rejected chunk.
    """

    total_input: int = 0
    total_valid: int = 0
    total_rejected: int = 0
    rejections: list[dict[str, Any]] = field(default_factory=list)


class ChunkValidator:
    """Validates chunks against configurable quality thresholds.

    Attributes:
        min_tokens: Minimum token count — reject below this.
    """

    def __init__(self, min_tokens: int = 50) -> None:
        self.min_tokens = min_tokens

    def validate(
        self, chunks: list[Chunk]
    ) -> tuple[list[Chunk], list[Chunk], ValidationReport]:
        """Validate a list of chunks.

        Rejection criteria:
            - token_count < min_tokens
            - content is empty or whitespace-only
            - metadata missing required fields (pmcid, section, chunk_type)

        Args:
            chunks: List of Chunk objects to validate.

        Returns:
            Tuple of (valid_chunks, rejected_chunks, validation_report).
        """
        valid: list[Chunk] = []
        rejected: list[Chunk] = []
        report = ValidationReport(total_input=len(chunks))

        for chunk in chunks:
            reasons = self._check_chunk(chunk)
            if reasons:
                rejected.append(chunk)
                report.rejections.append({
                    "chunk_id": chunk.chunk_id,
                    "reasons": reasons,
                })
                logger.debug(
                    "Rejected chunk %s: %s", chunk.chunk_id, ", ".join(reasons)
                )
            else:
                valid.append(chunk)

        report.total_valid = len(valid)
        report.total_rejected = len(rejected)
        return valid, rejected, report

    def _check_chunk(self, chunk: Chunk) -> list[str]:
        """Check a single chunk against quality criteria.

        Args:
            chunk: The chunk to validate.

        Returns:
            List of rejection reason strings (empty if valid).
        """
        reasons: list[str] = []

        # Content checks
        if not chunk.content or not chunk.content.strip():
            reasons.append("empty_content")

        # Token count check
        if chunk.token_count < self.min_tokens:
            reasons.append(f"below_min_tokens ({chunk.token_count} < {self.min_tokens})")

        # Metadata checks
        if not chunk.metadata.pmcid:
            reasons.append("missing_pmcid")
        if not chunk.metadata.section and not chunk.metadata.chunk_type:
            reasons.append("missing_section_and_type")
        if not chunk.metadata.chunk_type:
            reasons.append("missing_chunk_type")

        return reasons
