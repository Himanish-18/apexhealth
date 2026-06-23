"""
Tests for XML parser wrapper (preprocessing.xml_parser).
"""

from pathlib import Path
from preprocessing.xml_parser import XMLParser


def test_parse_valid_file(tmp_path: Path):
    xml_content = """
    <article article-type="research-article">
      <front>
        <article-meta>
          <article-id pub-id-type="pmc">123</article-id>
          <title-group><article-title>Valid Title</article-title></title-group>
        </article-meta>
      </front>
      <body><sec><title>Intro</title><p>Hello</p></sec></body>
    </article>
    """
    input_path = tmp_path / "123.xml"
    output_path = tmp_path / "123.json"
    invalid_dir = tmp_path / "invalid"
    
    input_path.write_text(xml_content, encoding="utf-8")
    
    parser = XMLParser(invalid_json_dir=invalid_dir)
    success = parser.parse_file(input_path, output_path)
    
    assert success
    assert output_path.exists()
    assert not (invalid_dir / "123.json").exists()


def test_parse_invalid_file_quarantined(tmp_path: Path):
    xml_content = """
    <article>
      <front>
        <article-meta>
          <!-- Missing Title and PMCID -->
        </article-meta>
      </front>
    </article>
    """
    input_path = tmp_path / "456.xml"
    output_path = tmp_path / "456.json"
    invalid_dir = tmp_path / "invalid"
    
    input_path.write_text(xml_content, encoding="utf-8")
    
    parser = XMLParser(invalid_json_dir=invalid_dir)
    success = parser.parse_file(input_path, output_path)
    
    assert not success
    assert not output_path.exists()
    assert (invalid_dir / "456.json").exists()
