from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional
import json
import re

from bs4 import BeautifulSoup
from lxml import etree


BLOCK_TAGS = {
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "li",
    "table",
    "figcaption",
    "caption",
}

@dataclass(frozen=True)
class TableEntry:
    key: str
    markdown: str


@dataclass
class TableRegistry:
    entries: dict[str, TableEntry] = field(default_factory=dict)
    entries_by_key: dict[str, TableEntry] = field(default_factory=dict)
    inserted: set[str] = field(default_factory=set)

    def add_entry(self, key: str, markdown: str, aliases: Iterable[str]) -> None:
        entry = TableEntry(key=key, markdown=markdown)
        self.entries_by_key[key] = entry
        for alias in aliases:
            alias_key = _normalize_table_ref(alias)
            if alias_key:
                self.entries[alias_key] = entry

    def resolve(self, ref: str) -> Optional[str]:
        alias_key = _normalize_table_ref(ref)
        if not alias_key:
            return None
        entry = self.entries.get(alias_key)
        if entry is None or entry.key in self.inserted:
            return None
        self.inserted.add(entry.key)
        return entry.markdown

    def mark_inserted(self, key: Optional[str]) -> None:
        if not key:
            return
        alias_key = _normalize_table_ref(key)
        entry = self.entries.get(alias_key)
        if entry is not None:
            self.inserted.add(entry.key)
        else:
            self.inserted.add(alias_key)

    def remaining_markdown(self) -> list[str]:
        return [
            entry.markdown
            for key, entry in self.entries_by_key.items()
            if key not in self.inserted
        ]


@dataclass(frozen=True)
class AbbreviationEntry:
    short: str
    long: str
    definition_count: int
    count: int


@dataclass(frozen=True)
class DocumentIdentifiers:
    pmid: Optional[str]
    doi: Optional[str]
    pmcid: Optional[str]
    neurostore: Optional[str]
    raw: dict

    @classmethod
    def from_path(cls, path: Path) -> "DocumentIdentifiers":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            pmid=_clean_identifier(data.get("pmid")),
            doi=_clean_identifier(data.get("doi")),
            pmcid=_clean_identifier(data.get("pmcid")),
            neurostore=_clean_identifier(data.get("neurostore")),
            raw=data,
        )


@dataclass(frozen=True)
class SourceDocument:
    hash_id: str
    provider: str
    path: Path
    table_paths: list[Path]
    identifiers: DocumentIdentifiers


def discover_documents(input_dir: Path, providers: Optional[Iterable[str]] = None) -> list[SourceDocument]:
    provider_set = {p.lower() for p in providers} if providers else None
    documents: list[SourceDocument] = []
    for hash_dir in sorted([p for p in input_dir.iterdir() if p.is_dir()]):
        identifiers_path = hash_dir / "identifiers.json"
        if identifiers_path.exists():
            identifiers = DocumentIdentifiers.from_path(identifiers_path)
        else:
            identifiers = DocumentIdentifiers(None, None, None, hash_dir.name, {})
        source_dir = hash_dir / "source"
        if not source_dir.exists():
            continue
        for provider_dir in sorted([p for p in source_dir.iterdir() if p.is_dir()]):
            provider_name = provider_dir.name.lower()
            if provider_set and provider_name not in provider_set:
                continue
            primary_files = _select_primary_files(provider_name, provider_dir)
            if not primary_files:
                continue
            table_paths = _collect_table_paths(provider_dir)
            for path in primary_files:
                documents.append(
                    SourceDocument(
                        hash_id=hash_dir.name,
                        provider=provider_name,
                        path=path,
                        table_paths=table_paths,
                        identifiers=identifiers,
                    )
                )
    return documents


def extract_markdown_from_path(
    path: Path, table_paths: Iterable[Path] = (), backend: str = "native"
) -> str:
    table_registry = _build_table_registry(table_paths)
    suffix = path.suffix.lower()
    if suffix in {".html", ".htm"}:
        if backend == "docling":
            blocks = _extract_markdown_from_docling(path)
        else:
            html = path.read_text(encoding="utf-8", errors="ignore")
            blocks = _extract_markdown_from_html(html, table_registry)
    elif suffix == ".pdf":
        if backend != "docling":
            raise ValueError("PDF extraction requires the docling backend.")
        blocks = _extract_markdown_from_docling(path)
    elif suffix == ".xml":
        blocks = _extract_markdown_from_xml(path, table_registry)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    if table_registry:
        remaining = table_registry.remaining_markdown()
        if remaining:
            blocks.append("## Tables (unreferenced)")
            blocks.extend(remaining)

    return "\n\n".join([block for block in blocks if block])


def extract_abbreviations(text: str) -> list[AbbreviationEntry]:
    text = text.strip()
    if not text:
        return []

    nlp = _get_abbreviation_nlp()
    doc = nlp(text)
    counts: dict[tuple[str, str], int] = {}
    for abrv in doc._.abbreviations:
        short = _normalize_abbrev(str(abrv))
        long = _normalize_abbrev(str(abrv._.long_form))
        if not short or not long:
            continue
        key = (short, long)
        counts[key] = counts.get(key, 0) + 1

    entries = []
    for (short, long), definition_count in sorted(
        counts.items(), key=lambda item: (item[0][0].lower(), item[0][1].lower())
    ):
        count = _count_abbrev_mentions(text, short)
        entries.append(
            AbbreviationEntry(
                short=short,
                long=long,
                definition_count=definition_count,
                count=count,
            )
        )
    return entries


def build_document_id(identifiers: DocumentIdentifiers, fallback: str) -> str:
    parts = []
    for value in (identifiers.pmid, identifiers.doi, identifiers.pmcid):
        if value:
            parts.append(_sanitize_identifier(value))
    if not parts:
        parts.append(_sanitize_identifier(identifiers.neurostore or fallback))
    return "-".join(parts)


def extract_tables_from_paths(table_paths: Iterable[Path]) -> list[str]:
    blocks: list[str] = []
    for table_path in sorted(table_paths):
        suffix = table_path.suffix.lower()
        if suffix in {".html", ".htm"}:
            html = table_path.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(html, "lxml")
            for table in soup.find_all("table"):
                markdown = html_table_to_markdown(table)
                if markdown:
                    blocks.append(markdown)
        elif suffix == ".xml":
            try:
                tree = etree.parse(str(table_path))
            except etree.XMLSyntaxError:
                continue
            table_blocks = _extract_tables_from_xml_tree(tree)
            blocks.extend(table_blocks)
    return blocks


def _build_table_registry(table_paths: Iterable[Path]) -> Optional[TableRegistry]:
    table_paths = list(table_paths)
    if not table_paths:
        return None

    registry = TableRegistry()
    for table_path in sorted(table_paths):
        suffix = table_path.suffix.lower()
        if suffix in {".html", ".htm"}:
            _register_html_tables(table_path, registry)
        elif suffix == ".xml":
            _register_xml_tables(table_path, registry)
    return registry


def html_table_to_markdown(table) -> str:
    thead = table.find("thead")
    tbody = table.find("tbody")
    header_rows = _parse_html_table_section(thead) if thead else []
    body_rows = _parse_html_table_section(tbody) if tbody else []

    if not header_rows and not body_rows:
        all_rows = _parse_html_table_section(table)
        if all_rows:
            header_rows = all_rows[:1]
            body_rows = all_rows[1:]

    headers = _merge_header_rows(header_rows, body_rows)
    return _build_markdown_table(headers, body_rows)


def _register_html_tables(table_path: Path, registry: TableRegistry) -> None:
    html = table_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")
    if not tables:
        return
    for index, table in enumerate(tables):
        markdown = html_table_to_markdown(table)
        if not markdown:
            continue
        key, aliases = _derive_table_key_from_html(table_path, table, index)
        registry.add_entry(key, markdown, aliases)


def _register_xml_tables(table_path: Path, registry: TableRegistry) -> None:
    try:
        tree = etree.parse(str(table_path))
    except etree.XMLSyntaxError:
        return
    root = tree.getroot()
    tables = []
    root_name = _local_name(root)
    if root_name in {"table", "table-wrap"}:
        tables = [root]
    else:
        tables = [node for node in root.iter() if _local_name(node) in {"table", "table-wrap"}]
    for index, table in enumerate(tables):
        table_id, markdown = _xml_table_to_markdown_with_id(table)
        if not markdown:
            continue
        key = table_id or _derive_table_key_from_path(table_path, index)
        aliases = _aliases_for_table_key(key)
        registry.add_entry(key, markdown, aliases)


def _extract_markdown_from_html(
    html: str, table_registry: Optional[TableRegistry]
) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    container = (
        soup.select_one('[data-article-body="true"]')
        or soup.select_one("article")
        or soup.select_one("main")
        or soup.body
        or soup
    )
    for tag in container.find_all(["script", "style", "nav", "aside", "footer", "header"]):
        tag.decompose()

    blocks: list[str] = []
    for element in container.find_all(list(BLOCK_TAGS)):
        if element.name != "table" and element.find_parent("table"):
            continue
        if element.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(element.name[1])
            text = _normalize_text(element.get_text(" ", strip=True))
            if text:
                blocks.append(f"{'#' * level} {text}")
        elif element.name == "table":
            markdown = html_table_to_markdown(element)
            if markdown:
                blocks.append(markdown)
                if table_registry is not None:
                    table_registry.mark_inserted(_normalize_table_ref(element.get("id")))
        elif element.name == "li":
            text = _normalize_text(element.get_text(" ", strip=True))
            if text:
                blocks.append(f"- {text}")
        else:
            text = _normalize_text(element.get_text(" ", strip=True))
            if text:
                blocks.append(text)
        if table_registry is not None:
            for ref in _extract_html_table_refs(element):
                table_markdown = table_registry.resolve(ref)
                if table_markdown:
                    blocks.append(table_markdown)
    return blocks


def _extract_markdown_from_xml(
    path: Path, table_registry: Optional[TableRegistry]
) -> list[str]:
    try:
        tree = etree.parse(str(path))
    except etree.XMLSyntaxError:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [_normalize_text(text)] if text else []

    root = tree.getroot()
    root_name = etree.QName(root).localname.lower()
    if root_name == "full-text-retrieval-response":
        return _extract_elsevier_xml(tree, table_registry)
    if root_name == "article":
        return _extract_jats_xml(tree, table_registry)
    return _extract_generic_xml(tree)


def _extract_markdown_from_docling(path: Path) -> list[str]:
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise ImportError(
            "Docling backend requested but docling is not installed. "
            "Install with: pip install '.[docling]'"
        ) from exc

    converter = DocumentConverter()
    result = converter.convert(str(path))
    doc = getattr(result, "document", result)
    markdown = _docling_export_markdown(doc)
    if markdown:
        return [markdown.strip()]

    text = _docling_export_text(doc)
    if text:
        return [_normalize_text(text)]
    return []


def _docling_export_markdown(doc) -> str:
    for attr in ("export_to_markdown", "to_markdown", "as_markdown"):
        if hasattr(doc, attr):
            return getattr(doc, attr)()
    return ""


def _docling_export_text(doc) -> str:
    for attr in ("export_to_text", "to_text", "as_text"):
        if hasattr(doc, attr):
            return getattr(doc, attr)()
    return ""


def _extract_elsevier_xml(
    tree: etree._ElementTree, table_registry: Optional[TableRegistry]
) -> list[str]:
    blocks: list[str] = []
    root = tree.getroot()

    abstract = _find_first_descendant(root, "abstract")
    if abstract is not None:
        abstract_text = _normalize_text(" ".join(abstract.itertext()))
        if abstract_text:
            blocks.append("## Abstract")
            blocks.append(abstract_text)

    body = _find_first_descendant(root, "body")
    if body is None:
        return blocks

    sections_container = _find_first_child(body, "sections")
    if sections_container is None:
        sections_container = body
    for section in sections_container:
        if _local_name(section) != "section":
            continue
        _walk_elsevier_section(section, 2, blocks, table_registry)
    return blocks


def _walk_elsevier_section(
    section, level: int, blocks: list[str], table_registry: Optional[TableRegistry]
) -> None:
    title = _first_text(section, "section-title")
    if title:
        blocks.append(f"{'#' * min(level, 6)} {title}")

    for child in section:
        name = _local_name(child)
        if name in {"para", "simple-para"}:
            text = _normalize_text(" ".join(child.itertext()))
            if text:
                blocks.append(text)
            if table_registry is not None:
                for refid in _extract_elsevier_table_refs(child):
                    table_markdown = table_registry.resolve(refid)
                    if table_markdown:
                        blocks.append(table_markdown)
        elif name == "table":
            markdown = cals_table_to_markdown(child)
            if markdown:
                blocks.append(markdown)
                if table_registry is not None:
                    table_registry.mark_inserted(_normalize_table_ref(child.get("id")))
        elif name == "section":
            _walk_elsevier_section(child, level + 1, blocks, table_registry)


def _extract_jats_xml(
    tree: etree._ElementTree, table_registry: Optional[TableRegistry]
) -> list[str]:
    blocks: list[str] = []
    root = tree.getroot()

    abstract = _find_first_descendant(root, "abstract")
    if abstract is not None:
        abstract_text = _normalize_text(" ".join(abstract.itertext()))
        if abstract_text:
            blocks.append("## Abstract")
            blocks.append(abstract_text)

    body = _find_first_descendant(root, "body")
    if body is None:
        return blocks

    for sec in body:
        if _local_name(sec) != "sec":
            continue
        _walk_jats_section(sec, 2, blocks, table_registry)
    return blocks


def _walk_jats_section(
    section, level: int, blocks: list[str], table_registry: Optional[TableRegistry]
) -> None:
    title = _first_text(section, "title")
    if title:
        blocks.append(f"{'#' * min(level, 6)} {title}")

    for child in section:
        name = _local_name(child)
        if name == "p":
            text = _normalize_text(_text_without_tables(child))
            if text:
                blocks.append(text)
            for table_wrap in [
                node for node in child.iter() if _local_name(node) == "table-wrap"
            ]:
                table_id, table_block = _table_wrap_to_markdown_with_id(table_wrap)
                if table_block:
                    blocks.append(table_block)
                if table_registry is not None:
                    table_registry.mark_inserted(table_id)
            if table_registry is not None:
                for refid in _extract_jats_table_refs(child):
                    table_markdown = table_registry.resolve(refid)
                    if table_markdown:
                        blocks.append(table_markdown)
        elif name == "table-wrap":
            table_id, table_block = _table_wrap_to_markdown_with_id(child)
            if table_block:
                blocks.append(table_block)
            if table_registry is not None:
                table_registry.mark_inserted(table_id)
        elif name == "sec":
            _walk_jats_section(child, level + 1, blocks, table_registry)


def _extract_generic_xml(tree: etree._ElementTree) -> list[str]:
    text = _normalize_text(" ".join(tree.getroot().itertext()))
    return [text] if text else []


def _extract_tables_from_xml_tree(tree: etree._ElementTree) -> list[str]:
    root = tree.getroot()
    root_name = _local_name(root)
    blocks: list[str] = []

    if root_name == "table":
        markdown = cals_table_to_markdown(root)
        if markdown:
            blocks.append(markdown)
        return blocks
    if root_name == "table-wrap":
        table_block = _table_wrap_to_markdown(root)
        if table_block:
            blocks.append(table_block)
        return blocks

    for table in [node for node in root.iter() if _local_name(node) == "table"]:
        markdown = cals_table_to_markdown(table)
        if markdown:
            blocks.append(markdown)
    return blocks


def _table_wrap_to_markdown(table_wrap) -> str:
    _, markdown = _table_wrap_to_markdown_with_id(table_wrap)
    return markdown


def _table_wrap_to_markdown_with_id(table_wrap) -> tuple[Optional[str], str]:
    label = _first_text(table_wrap, "label")
    caption = _first_text(table_wrap, "caption")
    table = _find_first_descendant(table_wrap, "table")
    if table is None:
        return table_wrap.get("id"), ""
    html = etree.tostring(table, encoding="unicode")
    soup = BeautifulSoup(html, "lxml")
    markdown = html_table_to_markdown(soup.find("table"))
    if not markdown:
        return table_wrap.get("id"), ""

    parts = []
    if label:
        parts.append(label)
    if caption:
        parts.append(caption)
    if parts:
        parts.append(markdown)
        return table_wrap.get("id"), "\n\n".join(parts)
    return table_wrap.get("id"), markdown


def cals_table_to_markdown(table) -> str:
    tgroup = _find_first_descendant(table, "tgroup")
    if tgroup is None:
        return ""

    colspecs = [
        child for child in tgroup.findall("./*") if _local_name(child) == "colspec"
    ]
    if colspecs:
        colnames = [col.get("colname") or f"col{idx + 1}" for idx, col in enumerate(colspecs)]
    else:
        cols = int(tgroup.get("cols", "0"))
        colnames = [f"col{idx + 1}" for idx in range(cols)]

    thead = _find_first_child(tgroup, "thead")
    tbody = _find_first_child(tgroup, "tbody")

    header_rows = _parse_cals_rows(thead, colnames) if thead is not None else []
    body_rows = _parse_cals_rows(tbody, colnames) if tbody is not None else []

    headers = _merge_header_rows(header_rows, body_rows)
    return _build_markdown_table(headers, body_rows)


def _xml_table_to_markdown_with_id(table) -> tuple[Optional[str], str]:
    name = _local_name(table)
    if name == "table-wrap":
        return _table_wrap_to_markdown_with_id(table)
    if name == "table":
        return table.get("id"), cals_table_to_markdown(table)
    return None, ""


def _derive_table_key_from_html(
    table_path: Path, table, index: int
) -> tuple[str, list[str]]:
    key = table.get("id") or ""
    if not key:
        caption = table.find("caption")
        if caption:
            number = _extract_table_number(caption.get_text(" ", strip=True))
            if number:
                key = f"table{number}"
    if not key:
        key = _derive_table_key_from_path(table_path, index)
    return key, _aliases_for_table_key(key)


def _derive_table_key_from_path(table_path: Path, index: int) -> str:
    number = _extract_table_number(table_path.stem)
    if number:
        return f"table{number}"
    return f"{table_path.stem}-{index + 1}"


def _aliases_for_table_key(key: str) -> list[str]:
    aliases = {key}
    norm = _normalize_table_ref(key)
    if norm:
        aliases.add(norm)
    number = _extract_table_number(key)
    if number:
        aliases.update(
            {
                number,
                f"table{number}",
                f"table {number}",
                f"tab{number}",
                f"tbl{number}",
            }
        )
    return list(aliases)


def _extract_table_number(text: str) -> Optional[str]:
    match = re.search(r"(?:table|tab|tbl)?\\s*(\\d+)", text, flags=re.I)
    if match:
        return match.group(1)
    return None


def _extract_html_table_refs(element) -> list[str]:
    refs: list[str] = []
    for link in element.find_all("a"):
        href = link.get("href", "")
        text = link.get_text(" ", strip=True)
        refs.extend(_extract_table_refs_from_href(href))
        refs.extend(_extract_table_refs_from_text(text))
        if link.get("id"):
            refs.append(link.get("id"))
    if not refs:
        refs.extend(_extract_table_refs_from_text(element.get_text(" ", strip=True)))
    seen = set()
    unique_refs = []
    for ref in refs:
        norm = _normalize_table_ref(ref)
        if norm and norm not in seen:
            seen.add(norm)
            unique_refs.append(ref)
    return unique_refs


def _extract_table_refs_from_href(href: str) -> list[str]:
    refs: list[str] = []
    if "#" in href:
        refs.append(href.split("#", 1)[1])
    match = re.search(r"/tables/(\\d+)", href)
    if match:
        refs.append(f"table{match.group(1)}")
        refs.append(match.group(1))
    return refs


def _extract_table_refs_from_text(text: str) -> list[str]:
    refs: list[str] = []
    for match in re.finditer(r"table\\s*(\\d+)", text, flags=re.I):
        number = match.group(1)
        refs.extend([f"table{number}", number, f"tab{number}", f"tbl{number}"])
    return refs


def _extract_elsevier_table_refs(element) -> list[str]:
    refs = []
    for node in element.iter():
        if _local_name(node) in {"float-anchor", "cross-ref"}:
            refid = node.get("refid")
            if refid:
                refs.append(refid)
    return refs


def _extract_jats_table_refs(element) -> list[str]:
    refs = []
    for node in element.iter():
        if _local_name(node) == "xref":
            if node.get("ref-type") in {"table", "tbl", None}:
                rid = node.get("rid")
                if rid:
                    refs.append(rid)
    return refs


def _parse_html_table_section(section) -> list[list[str]]:
    if section is None:
        return []

    rows = section.find_all("tr")
    matrix: list[list[str]] = []
    pending: dict[tuple[int, int], str] = {}

    for row_idx, row in enumerate(rows):
        row_cells: list[str] = []
        col_idx = 0
        while (row_idx, col_idx) in pending:
            row_cells.append(pending[(row_idx, col_idx)])
            col_idx += 1

        # Handle row/col spans so table columns stay aligned.
        for cell in row.find_all(["th", "td"], recursive=False):
            text = _normalize_text(cell.get_text(" ", strip=True))
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))
            for offset in range(colspan):
                row_cells.append(text)
                if rowspan > 1:
                    for extra in range(1, rowspan):
                        pending[(row_idx + extra, col_idx + offset)] = text
            col_idx += colspan
            while (row_idx, col_idx) in pending:
                row_cells.append(pending[(row_idx, col_idx)])
                col_idx += 1

        matrix.append(row_cells)
    return matrix


def _parse_cals_rows(section, colnames: list[str]) -> list[list[str]]:
    if section is None:
        return []

    rows = [child for child in section if _local_name(child) == "row"]
    matrix: list[list[str]] = []
    pending: dict[tuple[int, int], str] = {}
    max_cols = len(colnames)

    for row_idx, row in enumerate(rows):
        row_cells = ["" for _ in range(max_cols)]
        for col_idx in range(max_cols):
            if (row_idx, col_idx) in pending:
                row_cells[col_idx] = pending[(row_idx, col_idx)]

        entries = [child for child in row if _local_name(child) == "entry"]
        next_col = 0
        for entry in entries:
            text = _normalize_text(" ".join(entry.itertext()))
            start, end = _entry_span(entry, colnames, row_cells, next_col)
            for col_idx in range(start, end + 1):
                row_cells[col_idx] = text
                morerows = int(entry.get("morerows", "0"))
                if morerows:
                    for extra in range(1, morerows + 1):
                        pending[(row_idx + extra, col_idx)] = text
            next_col = end + 1

        matrix.append(row_cells)
    return matrix


def _entry_span(entry, colnames: list[str], row_cells: list[str], start_col: int) -> tuple[int, int]:
    name_start = entry.get("namest")
    name_end = entry.get("nameend")
    colname = entry.get("colname")

    if name_start and name_end:
        try:
            return colnames.index(name_start), colnames.index(name_end)
        except ValueError:
            pass
    if colname:
        try:
            idx = colnames.index(colname)
            return idx, idx
        except ValueError:
            pass

    col = start_col
    while col < len(row_cells) and row_cells[col]:
        col += 1
    return col, col


def _merge_header_rows(header_rows: list[list[str]], body_rows: list[list[str]]) -> list[str]:
    max_cols = 0
    for row in header_rows + body_rows:
        max_cols = max(max_cols, len(row))
    if max_cols == 0:
        return []

    headers: list[str] = []
    for col_idx in range(max_cols):
        parts = []
        for row in header_rows:
            if col_idx < len(row):
                text = row[col_idx].strip()
                if text and text not in parts:
                    parts.append(text)
        header = " / ".join(parts) if parts else ""
        headers.append(header or f"Column {col_idx + 1}")
    return headers


def _build_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not headers and not rows:
        return ""

    max_cols = len(headers)
    for row in rows:
        max_cols = max(max_cols, len(row))
    if max_cols == 0:
        return ""

    if not headers:
        headers = [f"Column {idx + 1}" for idx in range(max_cols)]
    elif len(headers) < max_cols:
        headers = headers + [f"Column {idx + 1}" for idx in range(len(headers), max_cols)]

    header_line = "| " + " | ".join(_escape_cell(cell) for cell in headers) + " |"
    divider = "| " + " | ".join(["---"] * max_cols) + " |"
    row_lines = []
    for row in rows:
        padded = row + [""] * (max_cols - len(row))
        row_lines.append("| " + " | ".join(_escape_cell(cell) for cell in padded) + " |")

    return "\n".join([header_line, divider] + row_lines)


def _select_primary_files(provider_name: str, provider_dir: Path) -> list[Path]:
    if provider_name == "elsevier":
        content = provider_dir / "content.xml"
        if content.exists():
            return [content]
    if provider_name == "pubget":
        article = provider_dir / "article.xml"
        if article.exists():
            return [article]

    candidates = (
        list(provider_dir.glob("*.html"))
        + list(provider_dir.glob("*.xml"))
        + list(provider_dir.glob("*.pdf"))
    )
    return sorted(candidates)


def _collect_table_paths(provider_dir: Path) -> list[Path]:
    tables_dir = provider_dir / "tables"
    if not tables_dir.exists():
        return []
    table_paths = list(tables_dir.rglob("*.html")) + list(tables_dir.rglob("*.xml"))
    return sorted(table_paths)


def _first_text(element, name: str) -> str:
    node = _find_first_child(element, name)
    if node is None:
        return ""
    return _normalize_text(" ".join(node.itertext()))


def _text_without_tables(element) -> str:
    clone = etree.fromstring(etree.tostring(element))
    etree.strip_elements(clone, "table-wrap", "table", with_tail=False)
    return " ".join(clone.itertext())


def _local_name(element) -> str:
    if not isinstance(getattr(element, "tag", None), str):
        return ""
    return etree.QName(element).localname.lower()


def _find_first_child(element, name: str):
    for child in element:
        if _local_name(child) == name:
            return child
    return None


def _find_first_descendant(element, name: str):
    for node in element.iter():
        if _local_name(node) == name:
            return node
    return None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _normalize_abbrev(text: str) -> str:
    return _normalize_text(text)


def _count_abbrev_mentions(text: str, short: str) -> int:
    if not short:
        return 0
    pattern = re.compile(rf"(?<!\\w){re.escape(short)}(?!\\w)")
    return len(pattern.findall(text))


@lru_cache(maxsize=1)
def _get_abbreviation_nlp():
    try:
        import spacy
        from scispacy.abbreviation import AbbreviationDetector  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Abbreviation extraction requires scispaCy. "
            "Install with: uv pip install '.[abbrev]'"
        ) from exc

    nlp = spacy.blank("en")
    if "abbreviation_detector" not in nlp.pipe_names:
        nlp.add_pipe("abbreviation_detector")
    return nlp


def _normalize_table_ref(ref: Optional[str]) -> str:
    if not ref:
        return ""
    return re.sub(r"[^a-z0-9]+", "", ref.lower())


def _escape_cell(text: str) -> str:
    return _normalize_text(text).replace("|", "\\|")


def _clean_identifier(value: object) -> Optional[str]:
    if value is None:
        return None
    value_str = str(value).strip()
    return value_str or None


def _sanitize_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)
