"""
Healthcare Knowledge Navigator — Preprocessing Module.

Handles extraction, canonicalization, and chunking of structured content
from PMC Open Access JATS XML files.
"""

from preprocessing.canonicalizer import CanonicalDocument, Canonicalizer, FigureSchema
from preprocessing.chunk_metadata import build_metadata, generate_metadata_header
from preprocessing.chunk_pipeline import ChunkPipeline, ChunkStats
from preprocessing.chunk_validator import ChunkValidator, ValidationReport
from preprocessing.chunker import Chunk, ChunkedDocument, ChunkMetadata, ChunkType
from preprocessing.parser_pipeline import ParseStats, ParserPipeline
from preprocessing.parser_utils import clean_text
from preprocessing.section_extractor import SectionExtractor, SectionSchema
from preprocessing.semantic_chunker import SemanticChunker
from preprocessing.table_chunker import TableChunker as ChunkTableChunker
from preprocessing.table_extractor import TableExtractor, TableSchema
from preprocessing.token_counter import TokenCounter, get_token_counter
from preprocessing.xml_parser import XMLParser

__all__ = [
    # Phase 2 — Parsing
    "Canonicalizer",
    "CanonicalDocument",
    "FigureSchema",
    "SectionExtractor",
    "SectionSchema",
    "TableExtractor",
    "TableSchema",
    "XMLParser",
    "ParserPipeline",
    "ParseStats",
    "clean_text",
    # Phase 3 — Chunking
    "Chunk",
    "ChunkedDocument",
    "ChunkMetadata",
    "ChunkType",
    "SemanticChunker",
    "ChunkTableChunker",
    "ChunkValidator",
    "ValidationReport",
    "ChunkPipeline",
    "ChunkStats",
    "TokenCounter",
    "get_token_counter",
    "build_metadata",
    "generate_metadata_header",
]

