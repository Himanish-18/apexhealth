"""
Healthcare Knowledge Navigator — Ingestion Module.

Handles downloading and ingesting PMC Open Access JATS XML files
into the local data store for further processing.

Components:
    - PMCClient: NCBI E-utilities API client for search and fetch.
    - Downloader: Resilient file downloader with retry and backoff.
    - MetadataExtractor: Extracts structured metadata from JATS XML.
    - XMLValidator: Validates XML files for structural integrity.
    - IngestionLogger: Structured logging with statistics tracking.
    - PMCIngestionPipeline: Pipeline orchestrator tying it all together.
"""

from ingestion.download_pmc import PMCIngestionPipeline
from ingestion.downloader import DownloadResult, Downloader
from ingestion.logger import IngestionLogger, IngestionStats
from ingestion.metadata_extractor import MetadataExtractor, PaperMetadata
from ingestion.pmc_api import PMCClient, PMCSearchResult
from ingestion.validator import ValidationResult, XMLValidator

__all__ = [
    "PMCIngestionPipeline",
    "PMCClient",
    "PMCSearchResult",
    "Downloader",
    "DownloadResult",
    "MetadataExtractor",
    "PaperMetadata",
    "XMLValidator",
    "ValidationResult",
    "IngestionLogger",
    "IngestionStats",
]
