"""
Tests for section extractor (preprocessing.section_extractor).
"""

from bs4 import BeautifulSoup
from preprocessing.section_extractor import SectionExtractor


def test_extract_sections():
    xml = """
    <article>
      <body>
        <sec>
          <title>Introduction</title>
          <p>This is the introduction.</p>
        </sec>
        <sec>
          <title>Methods</title>
          <p>Method main text.</p>
          <sec>
            <title>Participants</title>
            <p>100 participants.</p>
          </sec>
          <sec>
            <title>Procedure</title>
            <p>Did stuff.</p>
          </sec>
        </sec>
      </body>
    </article>
    """
    soup = BeautifulSoup(xml, "lxml-xml")
    body = soup.find("body")
    extractor = SectionExtractor()
    sections = extractor.extract_sections(body)

    assert len(sections) == 2
    
    sec1 = sections[0]
    assert sec1.title == "Introduction"
    assert sec1.content == "This is the introduction."
    assert len(sec1.subsections) == 0

    sec2 = sections[1]
    assert sec2.title == "Methods"
    assert sec2.content == "Method main text."
    assert len(sec2.subsections) == 2

    subsec1 = sec2.subsections[0]
    assert subsec1.title == "Participants"
    assert subsec1.content == "100 participants."

    subsec2 = sec2.subsections[1]
    assert subsec2.title == "Procedure"
    assert subsec2.content == "Did stuff."


def test_extract_sections_skips_tables():
    xml = """
    <article>
      <body>
        <sec>
          <title>Results</title>
          <p>Text before table.</p>
          <table-wrap>
            <caption>Table Caption</caption>
          </table-wrap>
          <p>Text after table.</p>
        </sec>
      </body>
    </article>
    """
    soup = BeautifulSoup(xml, "lxml-xml")
    body = soup.find("body")
    extractor = SectionExtractor()
    sections = extractor.extract_sections(body)

    assert len(sections) == 1
    sec = sections[0]
    # table-wrap is skipped, so caption doesn't end up in section content
    assert sec.content == "Text before table. Text after table."
