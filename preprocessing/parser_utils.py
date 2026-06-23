"""
Healthcare Knowledge Navigator — Parser Utilities.

Provides utility functions for robust text cleaning and normalization
during XML parsing.

Features:
- Whitespace normalization
- Unicode normalization (preserving biomedical/math symbols)
- Inline citation stripping using regex
- Safe text extraction from BeautifulSoup tags
"""

import re
import unicodedata
from typing import Any

# Regex to match common inline citations like [1], [1,2], [1-3], (1), (Author, 2020)
# This is a heuristic and might need tuning based on specific corpus characteristics.
# Matches:
#   [1]
#   [1, 2, 3]
#   [1-5]
#   (Author et al., 2020)
CITATION_PATTERN = re.compile(
    r"(?:\[[\d\s\,\-]+\])|(?:\(\s*[A-Za-z]+.*?(?:19|20)\d{2}\s*\))"
)


def normalize_whitespace(text: str) -> str:
    """
    Remove duplicate spaces, newlines, and tabs while preserving single paragraph breaks.

    Args:
        text: The input text.

    Returns:
        The text with normalized whitespace.
    """
    if not text:
        return ""

    # Replace multiple newlines with a single newline (to preserve paragraphs roughly if needed,
    # though usually we want to collapse it all to space for pure text chunks, unless it's structural).
    # We will collapse all whitespace (newlines, tabs) into single spaces.
    # If structural newlines are needed, they should be handled at the block level.
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters.
    Uses NFKC normalization but ensures critical biomedical symbols aren't destroyed.
    (NFKC can sometimes decompose things we want to keep, but it's generally safe for text.
     If specific Greek/Math symbols are mangled, NFKC might need to be NFC).

    Args:
        text: The raw unicode text.

    Returns:
        Normalized string.
    """
    if not text:
        return ""
    # NFC is safer for preserving Greek and math symbols exactly as they are
    # compared to NFKC which might decompose or replace them with ASCII equivalents.
    normalized = unicodedata.normalize("NFC", text)
    return normalized


def clean_citations(text: str) -> str:
    """
    Remove inline citation markers from the text.

    Args:
        text: Text potentially containing inline citations like [12].

    Returns:
        Text with citations removed and whitespace cleaned up around them.
    """
    if not text:
        return ""

    cleaned = CITATION_PATTERN.sub("", text)
    # Removing citations might leave double spaces before punctuation like "word  ."
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    # And double spaces generally
    cleaned = normalize_whitespace(cleaned)
    return cleaned


def clean_text(text: str) -> str:
    """
    Apply all standard cleaning: unicode, citations, and whitespace.
    """
    if not text:
        return ""
    text = normalize_unicode(text)
    text = clean_citations(text)
    text = normalize_whitespace(text)
    return text


def extract_text_from_element(element: Any) -> str:
    """
    Safely extract text from a BeautifulSoup tag, preserving math formulas as text
    if possible, and ignoring unwanted hidden elements.

    Args:
        element: A BeautifulSoup tag or NavigableString.

    Returns:
        The extracted and normalized text.
    """
    if element is None:
        return ""

    # element could be a string if it's already a NavigableString
    if hasattr(element, "get_text"):
        # We can extract text with a space separator to ensure words don't mash together
        # when tags are stripped (e.g., <p>Word1</p><p>Word2</p> -> Word1 Word2)
        raw_text = element.get_text(separator=" ", strip=True)
    else:
        raw_text = str(element)

    return clean_text(raw_text)
