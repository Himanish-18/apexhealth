"""
Healthcare Knowledge Navigator — Sparse BM25 Index.

Generates BM25 sparse vectors using the fastembed ``SparseTextEmbedding``
model.  Sparse vectors are stored alongside dense vectors in Qdrant to
enable hybrid (dense + BM25) retrieval.
"""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import models

logger = logging.getLogger(__name__)


class SparseEncoder:
    """Encodes text into BM25 sparse vectors using fastembed.

    The model is loaded lazily on first use so that importing the module
    does not trigger a heavyweight download.

    Attributes:
        model_name: Name of the fastembed sparse model.
    """

    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        """Initialize the sparse encoder.

        Args:
            model_name: fastembed sparse model identifier.
        """
        self.model_name = model_name
        self._model: Any = None

    def _load_model(self) -> None:
        """Lazy-load the SparseTextEmbedding model."""
        if self._model is None:
            from fastembed import SparseTextEmbedding

            logger.info("Loading sparse BM25 model: %s", self.model_name)
            self._model = SparseTextEmbedding(model_name=self.model_name)
            logger.info("Sparse BM25 model loaded.")

    def encode(self, texts: list[str]) -> list[models.SparseVector]:
        """Encode a list of texts into BM25 sparse vectors.

        Args:
            texts: List of text strings to encode.

        Returns:
            List of ``models.SparseVector`` objects with indices and values.
        """
        if not texts:
            return []

        self._load_model()

        sparse_embeddings = list(self._model.embed(texts))
        result: list[models.SparseVector] = []

        for emb in sparse_embeddings:
            result.append(
                models.SparseVector(
                    indices=emb.indices.tolist(),
                    values=emb.values.tolist(),
                )
            )

        return result

    def encode_single(self, text: str) -> models.SparseVector:
        """Encode a single text into a BM25 sparse vector.

        Args:
            text: Text string to encode.

        Returns:
            A ``models.SparseVector`` with indices and values.
        """
        results = self.encode([text])
        return results[0]
