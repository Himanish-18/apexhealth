"""
Healthcare Knowledge Navigator — Table Extractor.

Parses JATS XML <table-wrap> elements and converts them directly into
GitHub Flavored Markdown (GFM) format.
Associates each table with its caption and section context.
"""

from dataclasses import dataclass
from typing import Any

from bs4 import Tag

from preprocessing.parser_utils import extract_text_from_element


@dataclass
class TableSchema:
    """Represents an extracted table."""

    id: str
    caption: str
    markdown: str
    section: str = ""


class TableExtractor:
    """Extracts tables from JATS XML into Markdown format."""

    def extract_tables(self, soup: Any) -> list[TableSchema]:
        """
        Extract all <table-wrap> elements from the document.

        Args:
            soup: BeautifulSoup document object.

        Returns:
            List of TableSchema objects.
        """
        tables = []
        table_wraps = soup.find_all("table-wrap")

        for tw in table_wraps:
            table_id = tw.get("id", f"table_{len(tables)}")

            # Extract caption
            caption_tag = tw.find("caption")
            caption_text = extract_text_from_element(caption_tag) if caption_tag else ""

            # Extract section context
            # We look up the tree for the nearest <sec> and grab its title
            section_title = ""
            parent_sec = tw.find_parent("sec")
            if parent_sec:
                title_tag = parent_sec.find("title")
                if title_tag:
                    section_title = extract_text_from_element(title_tag)

            # Convert <table> to Markdown
            table_tag = tw.find("table")
            markdown_content = self._table_to_markdown(table_tag) if table_tag else ""

            # Only add if we actually extracted markdown
            if markdown_content:
                tables.append(
                    TableSchema(
                        id=table_id,
                        caption=caption_text,
                        markdown=markdown_content,
                        section=section_title,
                    )
                )

        return tables

    def _table_to_markdown(self, table_tag: Tag) -> str:
        """
        Convert a BeautifulSoup <table> tag to GitHub Flavored Markdown.

        Args:
            table_tag: BeautifulSoup <table> tag.

        Returns:
            Markdown string representation of the table.
        """
        md_rows = []

        # Process thead
        thead = table_tag.find("thead")
        if thead:
            headers = []
            # We assume a simple single-row thead for basic GFM conversion
            tr = thead.find("tr")
            if tr:
                for th in tr.find_all(["th", "td"]):
                    headers.append(extract_text_from_element(th).replace("|", "\\|"))
                
                if headers:
                    md_rows.append("| " + " | ".join(headers) + " |")
                    # Add separator row
                    md_rows.append("|" + "|".join(["---"] * len(headers)) + "|")

        # Process tbody
        tbody = table_tag.find("tbody") or table_tag
        trs = tbody.find_all("tr")
        
        # If there was no thead but we have rows, treat the first row as headers
        start_idx = 0
        if not md_rows and trs:
            headers = []
            for th in trs[0].find_all(["th", "td"]):
                headers.append(extract_text_from_element(th).replace("|", "\\|"))
            
            if headers:
                md_rows.append("| " + " | ".join(headers) + " |")
                md_rows.append("|" + "|".join(["---"] * len(headers)) + "|")
                start_idx = 1

        # Process data rows
        for i in range(start_idx, len(trs)):
            tr = trs[i]
            # Avoid picking up nested tables by limiting to direct children or simple find_all
            row_data = []
            for td in tr.find_all(["td", "th"], recursive=False):
                 row_data.append(extract_text_from_element(td).replace("|", "\\|"))
            
            if row_data:
                md_rows.append("| " + " | ".join(row_data) + " |")

        return "\n".join(md_rows)
