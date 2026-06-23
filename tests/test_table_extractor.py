"""
Tests for table extractor (preprocessing.table_extractor).
"""

from bs4 import BeautifulSoup
from preprocessing.table_extractor import TableExtractor


def test_extract_tables():
    xml = """
    <article>
      <body>
        <sec>
          <title>Results</title>
          <table-wrap id="T1">
            <caption>Table 1: Test Data</caption>
            <table>
              <thead>
                <tr><th>Name</th><th>Value</th></tr>
              </thead>
              <tbody>
                <tr><td>Test A</td><td>10</td></tr>
                <tr><td>Test B</td><td>20</td></tr>
              </tbody>
            </table>
          </table-wrap>
        </sec>
      </body>
    </article>
    """
    soup = BeautifulSoup(xml, "lxml-xml")
    extractor = TableExtractor()
    tables = extractor.extract_tables(soup)

    assert len(tables) == 1
    t = tables[0]
    assert t.id == "T1"
    assert t.caption == "Table 1: Test Data"
    assert t.section == "Results"
    
    expected_md = (
        "| Name | Value |\n"
        "|---|---|\n"
        "| Test A | 10 |\n"
        "| Test B | 20 |"
    )
    assert t.markdown == expected_md

def test_extract_tables_no_thead():
    xml = """
    <article>
      <table-wrap id="T2">
        <table>
          <tbody>
            <tr><td>H1</td><td>H2</td></tr>
            <tr><td>D1</td><td>D2</td></tr>
          </tbody>
        </table>
      </table-wrap>
    </article>
    """
    soup = BeautifulSoup(xml, "lxml-xml")
    extractor = TableExtractor()
    tables = extractor.extract_tables(soup)
    
    assert len(tables) == 1
    t = tables[0]
    
    expected_md = (
        "| H1 | H2 |\n"
        "|---|---|\n"
        "| D1 | D2 |"
    )
    assert t.markdown == expected_md
