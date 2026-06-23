"""
Tests for parser utilities (preprocessing.parser_utils).
"""

from bs4 import BeautifulSoup
from preprocessing.parser_utils import (
    clean_citations,
    extract_text_from_element,
    normalize_unicode,
    normalize_whitespace,
)


def test_normalize_whitespace():
    assert normalize_whitespace("This   is \n a \t test.") == "This is a test."
    assert normalize_whitespace("  Leading and trailing  ") == "Leading and trailing"
    assert normalize_whitespace(None) == ""
    assert normalize_whitespace("") == ""


def test_normalize_unicode():
    # Example of non-normalized unicode (e.g., e + accent vs é)
    text1 = "cafe\u0301"  # NFD
    text2 = "caf\u00e9"  # NFC
    assert normalize_unicode(text1) == text2


def test_clean_citations():
    text = "This is a sentence [1]. This is another [1,2, 3]. Third [1-5]."
    cleaned = clean_citations(text)
    assert cleaned == "This is a sentence. This is another. Third."

    text2 = "Smith (Smith et al., 2020) found that (Doe, 1999)."
    cleaned2 = clean_citations(text2)
    assert cleaned2 == "Smith found that."


def test_extract_text_from_element():
    html = "<p>This is <b>bold</b> and <i>italic</i>.</p>"
    soup = BeautifulSoup(html, "html.parser")
    # Tags removed, words joined with space and normalized
    text = extract_text_from_element(soup)
    assert text == "This is bold and italic."

    # Test math formula preservation
    math_html = "<disp-formula>E = mc^2</disp-formula>"
    math_soup = BeautifulSoup(math_html, "html.parser")
    assert extract_text_from_element(math_soup) == "E = mc^2"
