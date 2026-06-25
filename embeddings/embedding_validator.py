"""
Healthcare Knowledge Navigator — Embedding Validator.

Post-generation validation of every embedding vector. Ensures correct
dimensionality, numerical integrity (no NaN/Inf), unit L2 norm, and
non-empty content before writing to disk.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingValidationReport:
    """Report summarizing embedding validation results.

    Attributes:
        total_checked: Number of embeddings inspected.
        total_valid: Number that passed all checks.
        total_invalid: Number that failed at least one check.
        issues: List of dicts describing each invalid embedding.
    """

    total_checked: int = 0
    total_valid: int = 0
    total_invalid: int = 0
    issues: list[dict[str, Any]] = field(default_factory=list)


# Tolerance for unit-norm check: |‖v‖ − 1.0| < ε
_NORM_TOLERANCE: float = 1e-3


class EmbeddingValidator:
    """Validates embedding vectors against quality constraints.

    Checks performed per embedding:
        - Correct dimension matches expected model output
        - No NaN values
        - No Inf values
        - L2 norm is approximately 1.0 (unit vector)
        - Non-empty vector
    """

    def __init__(self, expected_dimension: int) -> None:
        """Initialize the validator.

        Args:
            expected_dimension: The expected embedding dimensionality
                                (e.g., 768 for MedCPT).
        """
        self.expected_dimension = expected_dimension

    def validate_embedding(
        self, chunk_id: str, embedding: list[float]
    ) -> list[str]:
        """Validate a single embedding vector.

        Args:
            chunk_id: Identifier of the chunk (for error reporting).
            embedding: The embedding vector as a list of floats.

        Returns:
            List of issue descriptions. Empty list means the embedding
            is valid.
        """
        issues: list[str] = []

        # 1. Non-empty
        if not embedding:
            issues.append("empty_embedding")
            return issues  # No further checks possible

        # 2. Correct dimension
        if len(embedding) != self.expected_dimension:
            issues.append(
                f"wrong_dimension (got {len(embedding)}, "
                f"expected {self.expected_dimension})"
            )

        # 3. No NaN
        if any(math.isnan(v) for v in embedding):
            issues.append("contains_nan")

        # 4. No Inf
        if any(math.isinf(v) for v in embedding):
            issues.append("contains_inf")

        # 5. Unit norm (only if no NaN/Inf to avoid math errors)
        if not issues or (
            "contains_nan" not in issues and "contains_inf" not in issues
        ):
            norm = math.sqrt(sum(v * v for v in embedding))
            if abs(norm - 1.0) > _NORM_TOLERANCE:
                issues.append(
                    f"non_unit_norm (norm={norm:.6f}, "
                    f"tolerance={_NORM_TOLERANCE})"
                )

        return issues

    def validate_document(
        self,
        chunks: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], EmbeddingValidationReport]:
        """Validate all embeddings in a document.

        Args:
            chunks: List of embedded chunk dicts, each with "chunk_id"
                    and "embedding" keys.

        Returns:
            Tuple of (valid_chunks, validation_report).
            Invalid chunks are excluded from the valid list.
        """
        report = EmbeddingValidationReport(total_checked=len(chunks))
        valid_chunks: list[dict[str, Any]] = []

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "unknown")
            embedding = chunk.get("embedding", [])

            issues = self.validate_embedding(chunk_id, embedding)

            if issues:
                report.total_invalid += 1
                report.issues.append({
                    "chunk_id": chunk_id,
                    "issues": issues,
                })
                logger.warning(
                    "Invalid embedding for chunk %s: %s",
                    chunk_id,
                    ", ".join(issues),
                )
            else:
                report.total_valid += 1
                valid_chunks.append(chunk)

        return valid_chunks, report
