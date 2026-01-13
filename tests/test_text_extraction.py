from pathlib import Path

from information_extraction.text.extractor import extract_markdown_from_path


ROOT = Path(__file__).resolve().parents[1]


def test_ace_html_extraction_includes_text_and_table() -> None:
    html_path = ROOT / "data/2NejviJMPeLn/source/ace/27761665.html"
    table_dir = ROOT / "data/2NejviJMPeLn/source/ace/tables"
    table_paths = list(table_dir.glob("*.html"))

    markdown = extract_markdown_from_path(html_path, table_paths)

    assert "Abstract" in markdown
    assert "Associations between long-term physical activity" in markdown
    assert "| Fitting window component |" in markdown
    assert "| ---" in markdown


def test_elsevier_xml_extraction_includes_text_and_table() -> None:
    xml_path = ROOT / "data/3AxBb8wzQbUH/source/elsevier/content.xml"
    table_dir = ROOT / "data/3AxBb8wzQbUH/source/elsevier/tables"
    table_paths = list(table_dir.glob("*.xml"))

    markdown = extract_markdown_from_path(xml_path, table_paths)

    assert "Introduction" in markdown
    assert "ventral visual stream are necessary for visual object recognition" in markdown
    assert "| Region |" in markdown


def test_pubget_xml_extraction_includes_text_and_table() -> None:
    xml_path = ROOT / "data/JVVjuCndcYD2/source/pubget/article.xml"
    table_dir = ROOT / "data/JVVjuCndcYD2/source/pubget/tables"
    table_paths = list(table_dir.glob("*.xml"))

    markdown = extract_markdown_from_path(xml_path, table_paths)

    assert "Global life expectancy has steadily and significantly increased" in markdown
    assert "| Normal |" in markdown
    assert "| MOAD |" in markdown
    assert "| OOAD |" in markdown
