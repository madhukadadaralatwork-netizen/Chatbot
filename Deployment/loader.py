"""
loader.py

Generic, domain-agnostic document loading. Replaces chunking_main.load_document
for docx/xlsx. The goal: adding a brand-new application's documents should
require zero changes to this file, regardless of what that application's
docs are about.

Two structural signals only -- no keywords, no domain assumptions:

  1. Section boundaries come from the document's own structure:
       - docx: Word heading styles ("Heading 1", "Heading 2", ...) define
         sections. Native Word tables are extracted in their actual
         document position (interleaved with paragraphs), not skipped --
         the original loader silently dropped every docx table, which is
         where most real "rules"/"spec" content usually lives.
       - xlsx: each sheet is a section (a spreadsheet's sheets ARE its
         author-defined structure already).

  2. Each emitted block is tagged is_table=True/False based on whether it's
     literally tabular (delimited rows), not by guessing what the table is
     *about*. This is the only "type" signal used downstream, and it works
     identically for a column-mapping table, a pricing table, an error-code
     table, or anything else -- because it doesn't try to know what the
     table means, just that it's structured reference data vs. prose.

Output: a flat list of Block(source, section, text, is_table) objects per
document. The chunker in retrieval_v2.py consumes these directly instead of
a single flattened string with inline "Source:"/"Sheet:" text markers.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

HEADING_STYLE_PREFIX = "Heading"


@dataclass
class Block:
    source: str
    section: str
    text: str
    is_table: bool = False


def _resolve_path(file_path):
    path = Path(file_path)
    if not path.exists():
        base_dir = Path(__file__).resolve().parent
        candidate = base_dir / file_path
        if candidate.exists():
            path = candidate
    return path


def _iter_block_items(document):
    """Yield paragraphs and tables in actual document order (python-docx
    exposes them separately by default, which loses interleaving)."""
    parent_elm = document.element.body
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def _table_to_text(table):
    rows = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def load_docx_blocks(file_path):
    path = _resolve_path(file_path)
    document = Document(str(path))
    source = os.path.basename(str(path))

    blocks = []
    current_section = "Introduction"
    paragraph_buffer = []

    def flush_paragraphs():
        if paragraph_buffer:
            text = "\n".join(paragraph_buffer)
            blocks.append(Block(source=source, section=current_section, text=text, is_table=False))
            paragraph_buffer.clear()

    for item in _iter_block_items(document):
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if not text:
                continue
            style_name = (item.style.name or "") if item.style else ""
            if style_name.startswith(HEADING_STYLE_PREFIX):
                flush_paragraphs()
                current_section = text
                continue
            paragraph_buffer.append(text)
        elif isinstance(item, Table):
            flush_paragraphs()
            table_text = _table_to_text(item)
            if table_text.strip():
                blocks.append(Block(source=source, section=current_section, text=table_text, is_table=True))

    flush_paragraphs()
    return blocks


def load_xlsx_blocks(file_path):
    path = _resolve_path(file_path)
    workbook = pd.ExcelFile(path)
    source = os.path.basename(str(path))

    blocks = []
    for sheet_name in workbook.sheet_names:
        df = workbook.parse(sheet_name)
        rows = []
        header_text = " | ".join(
            str(column).strip()
            for column in df.columns
            if pd.notna(column) and str(column).strip() and not str(column).startswith("Unnamed:")
        )
        if header_text:
            rows.append(header_text)
        for _, row in df.iterrows():
            text = " | ".join(str(value).strip() for value in row.tolist() if pd.notna(value) and str(value).strip())
            if text:
                rows.append(text)
        if rows:
            blocks.append(Block(source=source, section=sheet_name, text="\n".join(rows), is_table=True))

    return blocks


def load_document_blocks(file_path):
    path = _resolve_path(file_path)
    extension = os.path.splitext(str(path))[1].lower()
    if extension == ".docx":
        return load_docx_blocks(str(path))
    if extension == ".xlsx":
        return load_xlsx_blocks(str(path))
    raise ValueError(f"Unsupported file extension: {extension}")


def load_documents_blocks(file_paths):
    all_blocks = []
    for file_path in file_paths:
        all_blocks.extend(load_document_blocks(file_path))
    return all_blocks