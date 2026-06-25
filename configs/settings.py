"""
Healthcare Knowledge Navigator — Application Settings.

Centralized, type-safe configuration management using Pydantic Settings.
All values are loaded from environment variables (or .env file) with
sensible defaults for local development.
"""

from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Project Root ─────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables.

    Attributes:
        groq_api_key: API key for Groq LLM provider.
        qdrant_host: Hostname for the Qdrant vector database.
        qdrant_port: Port for the Qdrant REST API.
        embedding_model: Name of the embedding model to use.
        reranker_model: Name of the cross-encoder reranker model.
        collection_name: Qdrant collection name for medical documents.
        top_k: Number of candidates from initial retrieval.
        final_k: Number of results after re-ranking.
        log_level: Python logging level.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Provider ────────────────────────────────────────────────
    groq_api_key: str = ""

    # ── Vector Database ─────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # ── Models ──────────────────────────────────────────────────────
    embedding_model: str = "MedCPT"
    reranker_model: str = "bge-reranker-large"

    # ── Collection ──────────────────────────────────────────────────
    collection_name: str = "medical_documents"

    # ── Retrieval Parameters ────────────────────────────────────────
    top_k: int = 20
    final_k: int = 5

    # ── Logging ─────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Ingestion Pipeline ──────────────────────────────────────────
    max_downloads: int = 1000
    request_delay: float = 0.3
    max_retries: int = 3
    timeout: int = 20
    ncbi_api_key: str = ""
    ncbi_email: str = ""
    ncbi_tool: str = "HealthcareKnowledgeNavigator"

    # ── Parsing Pipeline ────────────────────────────────────────────
    max_parse_workers: int | None = None  # None = use CPU count

    # ── Chunking Pipeline ───────────────────────────────────────────
    max_chunk_workers: int | None = None  # None = use CPU count
    chunk_target_tokens: int = 512
    chunk_min_tokens: int = 50
    chunk_max_tokens: int = 600
    chunk_overlap_tokens: int = 75

    # ── Derived Paths ───────────────────────────────────────────────
    @property
    def data_dir(self) -> Path:
        """Root data directory."""
        return PROJECT_ROOT / "data"

    @property
    def raw_xml_dir(self) -> Path:
        """Directory for raw PMC JATS XML files."""
        return self.data_dir / "raw_xml"

    @property
    def parsed_json_dir(self) -> Path:
        """Directory for parsed document JSONs."""
        return self.data_dir / "parsed_json"

    @property
    def chunks_dir(self) -> Path:
        """Directory for chunked text segments."""
        return self.data_dir / "chunks"

    @property
    def metadata_dir(self) -> Path:
        """Directory for per-paper metadata JSON files."""
        return self.data_dir / "metadata"

    @property
    def parsing_logs_dir(self) -> Path:
        """Directory for parsing pipeline logs."""
        return self.data_dir / "logs" / "parsing"

    @property
    def invalid_json_dir(self) -> Path:
        """Directory for quarantined invalid JSON files."""
        return self.data_dir / "logs" / "invalid_json"

    @property
    def chunking_logs_dir(self) -> Path:
        """Directory for chunking pipeline logs and statistics."""
        return self.data_dir / "logs" / "chunking"

    @property
    def ingestion_logs_dir(self) -> Path:
        """Directory for ingestion pipeline logs."""
        return self.data_dir / "logs"

    @property
    def invalid_dir(self) -> Path:
        """Directory for quarantined invalid XML files."""
        return self.data_dir / "logs" / "invalid"

    @property
    def logs_dir(self) -> Path:
        """Directory for application logs."""
        return PROJECT_ROOT / "logs"

    @property
    def models_dir(self) -> Path:
        """Directory for downloaded model artifacts."""
        return PROJECT_ROOT / "models"

    @property
    def qdrant_url(self) -> str:
        """Fully-qualified Qdrant REST endpoint."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of application settings.

    Returns:
        Settings: The validated application settings.
    """
    return Settings()
