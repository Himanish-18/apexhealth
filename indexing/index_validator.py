"""
Healthcare Knowledge Navigator — Index Validator.

Post-indexing validation that verifies the Qdrant collection matches
the source embedding files: correct point count, vector dimensions,
payload completeness, and required metadata fields.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

# Fields that every payload must contain.
REQUIRED_PAYLOAD_KEYS: list[str] = [
    "chunk_id",
    "pmcid",
    "title",
    "section",
    "chunk_type",
    "token_count",
    "content",
]


@dataclass
class ValidationReport:
    """Report produced by index validation.

    Attributes:
        is_valid: True if all checks passed.
        total_points: Actual number of points in the collection.
        expected_points: Expected number of points from source.
        dimension_errors: Count of vectors with wrong dimensions.
        payload_errors: Count of points with incomplete payloads.
        missing_fields: Dict mapping point IDs to lists of missing fields.
        messages: List of human-readable validation messages.
    """

    is_valid: bool = True
    total_points: int = 0
    expected_points: int = 0
    dimension_errors: int = 0
    payload_errors: int = 0
    missing_fields: dict[int, list[str]] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)


class IndexValidator:
    """Validates the indexed Qdrant collection against source data.

    Performs lightweight sampling to verify vector dimensions and
    payload completeness without scanning every point.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        dense_vector_name: str = "dense",
        expected_dimension: int = 768,
    ) -> None:
        """Initialize the index validator.

        Args:
            client: Active QdrantClient instance.
            collection_name: Name of the collection to validate.
            dense_vector_name: Named vector key for dense embeddings.
            expected_dimension: Expected dimensionality of dense vectors.
        """
        self._client = client
        self.collection_name = collection_name
        self.dense_vector_name = dense_vector_name
        self.expected_dimension = expected_dimension

    def validate_collection(
        self,
        expected_count: int | None = None,
        sample_size: int = 100,
    ) -> ValidationReport:
        """Run validation checks against the collection.

        Checks:
            1. Point count matches expected count (if provided)
            2. Sampled vectors have the correct dimension
            3. Sampled payloads contain all required fields

        Args:
            expected_count: Expected number of points (None skips count check).
            sample_size: Number of points to sample for detailed checks.

        Returns:
            ValidationReport with all findings.
        """
        report = ValidationReport()

        # 1. Get collection info for point count
        try:
            info = self._client.get_collection(self.collection_name)
            report.total_points = info.points_count or 0
        except Exception as e:
            report.is_valid = False
            report.messages.append(f"Cannot access collection: {e}")
            return report

        # 2. Check point count
        if expected_count is not None:
            report.expected_points = expected_count
            if report.total_points != expected_count:
                report.is_valid = False
                report.messages.append(
                    f"Point count mismatch: expected {expected_count}, "
                    f"got {report.total_points}."
                )
            else:
                report.messages.append(
                    f"Point count OK: {report.total_points}"
                )
        else:
            report.expected_points = report.total_points
            report.messages.append(
                f"Collection has {report.total_points} points (no expected count given)."
            )

        # 3. Sample points for dimension and payload checks
        if report.total_points == 0:
            report.messages.append("No points to validate.")
            return report

        try:
            sampled_points = self._sample_points(sample_size)
        except Exception as e:
            report.messages.append(f"Sampling failed: {e}")
            report.is_valid = False
            return report

        for point in sampled_points:
            # Check vector dimensions
            if hasattr(point, "vector") and point.vector:
                dense_vec = point.vector.get(self.dense_vector_name)
                if dense_vec is not None and len(dense_vec) != self.expected_dimension:
                    report.dimension_errors += 1
                    report.is_valid = False

            # Check payload completeness
            if point.payload:
                missing = [
                    key for key in REQUIRED_PAYLOAD_KEYS
                    if key not in point.payload
                    or point.payload[key] is None
                    or point.payload[key] == ""
                ]
                if missing:
                    report.payload_errors += 1
                    report.missing_fields[point.id] = missing
                    report.is_valid = False

        if report.dimension_errors:
            report.messages.append(
                f"Dimension errors: {report.dimension_errors}/{len(sampled_points)} sampled."
            )
        else:
            report.messages.append("Vector dimensions OK.")

        if report.payload_errors:
            report.messages.append(
                f"Payload errors: {report.payload_errors}/{len(sampled_points)} sampled."
            )
        else:
            report.messages.append("Payload completeness OK.")

        return report

    def _sample_points(self, limit: int) -> list[Any]:
        """Sample points from the collection with vectors and payloads.

        Args:
            limit: Maximum number of points to sample.

        Returns:
            List of Record objects with vectors and payloads.
        """
        points, _ = self._client.scroll(
            collection_name=self.collection_name,
            limit=min(limit, 100),
            with_payload=True,
            with_vectors=[self.dense_vector_name],
        )
        return points
