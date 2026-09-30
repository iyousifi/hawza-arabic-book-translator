"""File I/O: read Arabic source files, normalize text, chunk classical discourse, and write translated output."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from translator.normalizer import (
    extract_text_from_pdf,
    normalize_arabic_text,
)


from translator.slicing import slice_text_by_pages


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def read_file(
    path: str,
    remove_headers_footers: bool = True,
    separate_footnotes: bool = True,
    exclude_margin_rubrics: bool = True,
    skip_pages: int = 0,
    take_pages: Optional[int] = None,
) -> str:
    """Read an input file and return its clean, normalized text content.

    Supported formats: .txt, .pdf, .docx
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".pdf":
        return extract_text_from_pdf(
            path=p,
            remove_headers_footers=remove_headers_footers,
            separate_footnotes=separate_footnotes,
            exclude_margin_rubrics=exclude_margin_rubrics,
            skip_pages=skip_pages,
            take_pages=take_pages,
        )

    if suffix in (".txt", ".md"):
        raw = _read_txt(p)
    elif suffix == ".docx":
        raw = _read_docx(p)
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}'. Supported: .txt, .md, .pdf, .docx"
        )

    norm = normalize_arabic_text(raw)
    if skip_pages > 0 or take_pages is not None:
        norm = slice_text_by_pages(norm, skip_pages=skip_pages, take_pages=take_pages)
    return norm


def _read_txt(path: Path) -> str:
    """Read a plain text file (UTF-8)."""
    return path.read_text(encoding="utf-8")


def _read_docx(path: Path) -> str:
    """Extract text from a DOCX file using python-docx."""
    try:
        from docx import Document
    except ImportError:
        raise ImportError(
            "python-docx is required to read DOCX files. "
            "Install it with: pip install python-docx"
        )

    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_output(text: str, path: str) -> None:
    """Write the translated text to the output file (UTF-8)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Default output path helper
# ---------------------------------------------------------------------------

def default_output_path(
    input_path: str,
    lang: str = "en",
    slice_tag: Optional[str] = None,
    bilingual: bool = False,
) -> str:
    """Generate a default output path from input path, language, and optional slice tag.

    e.g.  book.pdf, lang="da", slice_tag="p10-25", bilingual=True → book_p10-25_bilingual_da.md
    """
    p = Path(input_path)
    lang_suffix = f"_{lang.lower()}" if lang.lower() != "en" else ""
    tag_part = f"_{slice_tag}" if slice_tag else ""
    kind = "_bilingual" if bilingual else "_translated"
    return str(p.with_name(f"{p.stem}{tag_part}{kind}{lang_suffix}.md"))


def default_audit_path(output_path: str) -> str:
    """Generate a default audit report path corresponding to a translation output path.

    e.g.  book_translated_da.md  →  book_translated_da_audit.md
    """
    p = Path(output_path)
    return str(p.with_name(f"{p.stem}_audit.md"))


def default_pdf_path(output_path: str) -> str:
    """Generate a default PDF path corresponding to a translation output path.

    e.g.  book_translated_da.md  →  book_translated_da.pdf
    """
    p = Path(output_path)
    return str(p.with_suffix(".pdf"))



# ---------------------------------------------------------------------------
# Text chunking (re-exported from translator.chunker)
# ---------------------------------------------------------------------------

from translator.chunker import (
    TextChunk,
    chunk_classical_arabic,
    chunk_text,
    count_tokens,
)

