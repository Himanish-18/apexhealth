"""
Healthcare Knowledge Navigator — Embedding Model Interface.

Defines the abstract embedding model contract and the concrete MedCPT
implementation. Designed for swappability — future models (PubMedBERT,
BGE-M3, E5-large-v2) subclass ``EmbeddingModel`` without changing any
pipeline logic.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import torch

from embeddings.device_manager import DeviceInfo

logger = logging.getLogger(__name__)


class EmbeddingModel(ABC):
    """Abstract interface for dense text embedding models.

    All concrete implementations must provide:
        - ``model_name`` property
        - ``embedding_dimension`` property
        - ``encode()`` method returning L2-normalized float32 numpy arrays
        - ``to_device()`` to move the model to a specific compute device
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the canonical model identifier (e.g., 'MedCPT')."""

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the output embedding dimensionality."""

    @abstractmethod
    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of texts into normalized dense vectors.

        Args:
            texts: List of text strings to encode.
            batch_size: Number of texts to encode per forward pass.

        Returns:
            numpy array of shape (N, embedding_dimension) with L2 unit norm.
        """

    @abstractmethod
    def to_device(self, device: torch.device) -> None:
        """Move the model to the specified compute device.

        Args:
            device: Target torch device (cuda, mps, or cpu).
        """


class MedCPTEmbeddingModel(EmbeddingModel):
    """MedCPT Article Encoder for biomedical document embeddings.

    Wraps ``ncbi/MedCPT-Article-Encoder`` via sentence-transformers.
    Produces 768-dimensional L2-normalized embeddings optimized for
    biomedical literature retrieval.

    Attributes:
        _model: The underlying SentenceTransformer model.
        _device_info: Information about the active compute device.
    """

    EMBEDDING_DIM: int = 768

    def __init__(
        self,
        model_name_or_path: str = "ncbi/MedCPT-Article-Encoder",
        device_info: DeviceInfo | None = None,
    ) -> None:
        """Initialize the MedCPT embedding model.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            device_info: DeviceInfo for hardware placement. If None,
                         defaults to CPU.
        """
        self._model_id = model_name_or_path
        self._device_info = device_info

        # Deferred import to avoid pulling in the heavy transformers
        # dependency chain at module import time.
        from sentence_transformers import SentenceTransformer

        # Set deterministic behavior for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        device_str = device_info.device_type if device_info else "cpu"

        logger.info(
            "Loading embedding model: %s on %s",
            model_name_or_path,
            device_str,
        )

        self._model = SentenceTransformer(
            model_name_or_path,
            device=device_str,
        )

        logger.info(
            "Model loaded — dimension: %d, device: %s",
            self.embedding_dimension,
            device_str,
        )

    @property
    def model_name(self) -> str:
        """Return the canonical model name."""
        return "MedCPT"

    @property
    def embedding_dimension(self) -> int:
        """Return the output embedding dimensionality (768 for MedCPT)."""
        dim = self._model.get_sentence_embedding_dimension()
        return dim if dim is not None else self.EMBEDDING_DIM

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts into L2-normalized dense vectors.

        Args:
            texts: List of text strings to encode.
            batch_size: Number of texts per forward pass.

        Returns:
            numpy array of shape (len(texts), 768) with unit L2 norm.
        """
        if not texts:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2 normalization
            convert_to_numpy=True,
        )

        # Ensure float32 for consistent output
        embeddings = embeddings.astype(np.float32)

        return embeddings

    def to_device(self, device: torch.device) -> None:
        """Move the model to a different compute device.

        Args:
            device: Target torch device.
        """
        device_str = str(device)
        self._model = self._model.to(device_str)
        logger.info("Model moved to device: %s", device_str)
