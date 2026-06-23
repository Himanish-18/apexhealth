"""
Tests for canonicalizer (preprocessing.canonicalizer).
"""

from bs4 import BeautifulSoup
from preprocessing.canonicalizer import Canonicalizer


def test_canonicalizer():
    xml = """
    <article article-type="research-article">
      <front>
        <article-meta>
          <article-id pub-id-type="pmc">123456</article-id>
          <title-group><article-title>Test Title</article-title></title-group>
          <abstract><p>Test Abstract</p></abstract>
        </article-meta>
      </front>
      <body>
        <sec>
          <title>Intro</title>
          <p>Intro text.</p>
        </sec>
      </body>
      <back>
        <ref-list>
          <ref>Ref 1</ref>
          <ref>Ref 2</ref>
        </ref-list>
      </back>
    </article>
    """
    canonicalizer = Canonicalizer()
    doc = canonicalizer.process_document("PMC123456", xml)

    assert doc.metadata["pmcid"] == "PMC123456"
    assert doc.metadata["title"] == "Test Title"
    assert doc.abstract == "Test Abstract"
    assert len(doc.sections) == 1
    assert doc.sections[0].title == "Intro"
    assert doc.sections[0].content == "Intro text."
    assert len(doc.references) == 2
    assert doc.references[0] == "Ref 1"
    assert doc.references[1] == "Ref 2"
