"""
Healthcare Knowledge Navigator — Application Constants.

Immutable, application-wide constants. These do NOT change based on
environment — use ``configs.settings`` for anything configurable.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectMeta:
    """Immutable project metadata."""

    name: str = "Healthcare Knowledge Navigator"
    version: str = "0.1.0"
    description: str = (
        "A Healthcare RAG Assistant that retrieves evidence from "
        "PMC Open Access papers and generates evidence-grounded "
        "medical answers with citations and confidence scores."
    )
    author: str = "HKN Team"


@dataclass(frozen=True)
class LLMDefaults:
    """Default LLM configuration values."""

    provider: str = "groq"
    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.1
    max_tokens: int = 4096
    top_p: float = 0.95


@dataclass(frozen=True)
class RetrievalDefaults:
    """Default retrieval pipeline parameters."""

    # Reciprocal Rank Fusion constant
    rrf_k: int = 60
    # BM25 parameters
    bm25_b: float = 0.75
    bm25_k1: float = 1.5
    # Context assembly
    max_context_tokens: int = 6000


@dataclass(frozen=True)
class ChunkingDefaults:
    """Default chunking parameters for section-aware splitting."""

    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 100


# ── Singleton Instances ──────────────────────────────────────────────
PROJECT_META = ProjectMeta()
LLM_DEFAULTS = LLMDefaults()
RETRIEVAL_DEFAULTS = RetrievalDefaults()
CHUNKING_DEFAULTS = ChunkingDefaults()

# ── API Versioning ───────────────────────────────────────────────────
API_V1_PREFIX: str = "/api/v1"
