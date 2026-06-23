"""
Healthcare Knowledge Navigator — Preprocessing Module.

Handles extraction and canonicalization of structured content from
PMC Open Access JATS XML files.
"""

from preprocessing.canonicalizer import CanonicalDocument, Canonicalizer, FigureSchema
from preprocessing.parser_pipeline import ParseStats, ParserPipeline
from preprocessing.parser_utils import clean_text
from preprocessing.section_extractor import SectionExtractor, SectionSchema
from preprocessing.table_extractor import TableExtractor, TableSchema
from preprocessing.xml_parser import XMLParser

__all__ = [
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
]
