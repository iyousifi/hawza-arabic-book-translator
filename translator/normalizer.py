"""Smart Arabic Text Normalizer and Layout-Aware Extraction for Classical Scholastic Texts."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None


# ---------------------------------------------------------------------------
# Unicode Character Sets & Patterns
# ---------------------------------------------------------------------------

# Arabic diacritics (Tashkeel / Harakat)
_HARAKAT_RE = re.compile(
    r"[\u0617-\u061A\u064B-\u0652\u0656-\u065F\u0670]"
)

# Invisible directional controls and zero-width artifacts
_BIDI_CONTROLS_RE = re.compile(
    r"[\u200E\u200F\u202A-\u202E\u2066-\u2069\uFEFF\u200B\u200C\u200D]"
)

# Arabic Tatweel / Kashida (decorative elongation)
_TATWEEL_RE = re.compile(r"\u0640+")

# Footnote line rules (e.g. ________ or --------)
_FOOTNOTE_RULE_RE = re.compile(r"^[\s_—–-]{3,}\s*$")

# Footnote markers, e.g. (1), (١), [1], [١], (*), at start or end of block
_FOOTNOTE_MARKER_RE = re.compile(r"([(\[]\s*[\d٠-٩*†]+\s*[)\]]|\*|\b\d+[\-.)])")

# Pure page number pattern (Western or Eastern Arabic digits)
_PAGE_NUM_RE = re.compile(r"^\s*[-–—]?\s*[\d٠-٩]+\s*[-–—]?\s*$")


# ---------------------------------------------------------------------------
# Reversal (Bidi / Visual RTL) Detection & Repair
# ---------------------------------------------------------------------------

def is_arabic_text_reversed(text: str) -> bool:
    """Detect if Arabic text has been extracted in reversed (visual LTR) order."""
    words = re.findall(r"[\u0600-\u06FF]+", text)
    if not words:
        return False

    # 1. Taa Marbutah (ة) at the start of a word is grammatically impossible in Arabic
    if any(w.startswith("ة") for w in words):
        return True

    # 2. Definite article 'ال' check vs reversed 'لا' at word ends
    starts_al = sum(1 for w in words if w.startswith("ال"))
    ends_la = sum(1 for w in words if w.endswith("لا"))

    if ends_la > starts_al and ends_la > 0:
        return True

    return False


def fix_reversed_arabic_line(line: str) -> str:
    """Reverse a line back to logical RTL order and fix bracket/digit orientation."""
    rev = line[::-1]
    # Invert paired punctuation back to correct logical orientation
    bracket_swap = str.maketrans("()[]{}«»<>", ")(][}{»«><")
    rev = rev.translate(bracket_swap)
    # Digits are already written LTR, so line[::-1] flipped them - flip them back
    rev = re.sub(r"[\d٠-٩]+", lambda m: m.group(0)[::-1], rev)
    return rev


def detect_and_fix_reversed_arabic(text: str) -> str:
    """Detect if an Arabic text is reversed and restore it to logical reading order."""
    if not text:
        return ""

    if is_arabic_text_reversed(text):
        lines = text.split("\n")
        fixed_lines = [fix_reversed_arabic_line(l) for l in lines]
        return "\n".join(fixed_lines)

    return text


# ---------------------------------------------------------------------------
# Text Normalization
# ---------------------------------------------------------------------------

def normalize_arabic_text(
    text: str,
    strip_harakat: bool = False,
    standardize_digits: bool = False,
    fix_reversals: bool = True,
) -> str:
    """Normalize Arabic text extracted from digital files.

    Operations:
    1. Resolves Arabic Presentation Forms (A & B) to standard canonical Arabic via NFKD.
    2. Detects and corrects reversed (visual LTR) Arabic order.
    3. Strips calligraphic Tatweel/Kashida (ـ).
    4. Cleans invisible Bidi marks and zero-width spaces.
    5. Recomposes canonical diacritics via NFC.
    6. Normalizes Arabic punctuation spacing and line breaks.
    7. Optionally strips decorative Tashkeel or standardizes numerals.
    """
    if not text:
        return ""

    # Step 1: Decompose presentation forms (e.g. ﻼ \uFEFB, ﷲ \uFDF2) into standard Arabic
    decomposed = unicodedata.normalize("NFKD", text)

    # Step 2: Detect and correct reversed Arabic text if present
    if fix_reversals:
        decomposed = detect_and_fix_reversed_arabic(decomposed)

    # Step 3: Strip tatweel / kashida
    cleaned = _TATWEEL_RE.sub("", decomposed)

    # Step 4: Strip invisible Bidi directional markers and control characters
    cleaned = _BIDI_CONTROLS_RE.sub("", cleaned)
    cleaned = cleaned.replace("\u00a0", " ")  # Non-breaking space

    # Step 5: Re-compose to NFC canonical Unicode
    recomposed = unicodedata.normalize("NFC", cleaned)

    # Recompose decomposed Alef/Waw/Yeh + combining hamza/madda to standard precomposed letters
    recomposed = (
        recomposed
        .replace("\u0627\u0655", "\u0625")
        .replace("\u0655\u0627", "\u0625")
        .replace("\u0627\u0654", "\u0623")
        .replace("\u0654\u0627", "\u0623")
        .replace("\u0627\u0653", "\u0622")
        .replace("\u0653\u0627", "\u0622")
        .replace("\u0648\u0654", "\u0624")
        .replace("\u064A\u0654", "\u0626")
        .replace("\u0649\u0654", "\u0626")
    )

    # Step 6: Optional removal of harakat (if requested for search/matching)
    if strip_harakat:
        recomposed = _HARAKAT_RE.sub("", recomposed)

    # Step 7: Optional numeral standardization (Eastern Arabic ٠-٩ to Western 0-9)
    if standardize_digits:
        eastern_to_western = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
        recomposed = recomposed.translate(eastern_to_western)

    # Step 8: Clean whitespace and line endings
    recomposed = recomposed.replace("\r\n", "\n").replace("\r", "\n")
    recomposed = re.sub(r"[ \t]+", " ", recomposed)
    recomposed = re.sub(r"\n{3,}", "\n\n", recomposed)

    return recomposed.strip()


# ---------------------------------------------------------------------------
# Layout-Aware Block Extraction
# ---------------------------------------------------------------------------

@dataclass
class PageBlock:
    """Represents a text block with spatial coordinates on a page."""
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    is_footnote: bool = False
    is_header_or_footer: bool = False


def classify_page_blocks(
    blocks: List[Tuple[float, float, float, float, str, int, int]],
    page_width: float,
    page_height: float,
    remove_headers_footers: bool = True,
    separate_footnotes: bool = True,
) -> Tuple[List[str], List[str]]:
    """Classify extracted PDF blocks into body text and footnotes.

    Parameters
    ----------
    blocks:
        List of PyMuPDF text blocks (x0, y0, x1, y1, text, block_no, block_type).
    page_width:
        Width of the page.
    page_height:
        Height of the page.
    remove_headers_footers:
        Whether to filter out running headers (<7.5% page height) and solitary footers (>92.5%).
    separate_footnotes:
        Whether to identify and separate bottom commentary/footnotes.

    Returns
    -------
    Tuple[List[str], List[str]]
        (body_blocks, footnote_blocks)
    """
    classified: List[PageBlock] = []

    for b in blocks:
        # b[6] is block_type: 0 = text, 1 = image
        if len(b) <= 6 or b[6] != 0:
            continue

        raw_text = b[4].strip()
        if not raw_text:
            continue

        norm_text = normalize_arabic_text(raw_text)
        if not norm_text:
            continue

        x0, y0, x1, y1 = b[0], b[1], b[2], b[3]

        block = PageBlock(
            text=norm_text,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
        )

        # 1. Header detection: top 7.5% of page
        if remove_headers_footers and y1 < page_height * 0.075:
            block.is_header_or_footer = True

        # 2. Footer detection: bottom 7.5% of page and pure page number
        if remove_headers_footers and y0 > page_height * 0.925 and _PAGE_NUM_RE.match(norm_text):
            block.is_header_or_footer = True

        # 3. Footnote line rule
        if _FOOTNOTE_RULE_RE.match(norm_text):
            continue  # omit separator rule line itself

        classified.append(block)

    # Sort blocks: For RTL texts, right-hand columns come before left-hand columns
    is_multi_column = _detect_multi_column(classified, page_width)

    if is_multi_column:
        # Multi-column sorting: right column (higher x0) first, then by y0
        classified.sort(key=lambda blk: (0 if blk.x0 > page_width * 0.45 else 1, blk.y0))
    else:
        # Single column: top-to-bottom
        classified.sort(key=lambda blk: blk.y0)

    # Footnote boundary detection:
    # Footnotes in classical Hawza books appear at the bottom 35% of the page
    # and typically begin with (١) or footnote markers.
    footnote_started = False
    body_blocks: List[str] = []
    footnote_blocks: List[str] = []

    for blk in classified:
        if blk.is_header_or_footer:
            continue

        if separate_footnotes:
            in_bottom_zone = blk.y0 > page_height * 0.65
            has_marker = bool(
                _FOOTNOTE_MARKER_RE.search(blk.text[:20])
                or _FOOTNOTE_MARKER_RE.search(blk.text[-20:])
            )

            if in_bottom_zone and (has_marker or footnote_started):
                footnote_started = True
                blk.is_footnote = True

        if blk.is_footnote:
            footnote_blocks.append(blk.text)
        else:
            body_blocks.append(blk.text)

    return body_blocks, footnote_blocks


def _detect_multi_column(blocks: List[PageBlock], page_width: float) -> bool:
    """Detect if page blocks form a 2-column layout."""
    if len(blocks) < 4:
        return False

    midpoint = page_width * 0.5
    has_right = any(b.x0 > midpoint and b.x1 > page_width * 0.7 for b in blocks)
    has_left = any(b.x1 < midpoint and b.x0 < page_width * 0.3 for b in blocks)
    has_spanning = any(b.x0 < page_width * 0.25 and b.x1 > page_width * 0.75 for b in blocks)

    return has_right and has_left and not has_spanning


# ---------------------------------------------------------------------------
# PDF Reader using Smart Normalization
# ---------------------------------------------------------------------------

def extract_text_from_pdf(
    path: Path | str,
    remove_headers_footers: bool = True,
    separate_footnotes: bool = True,
    skip_pages: int = 0,
    take_pages: Optional[int] = None,
) -> str:
    """Extract clean, normalized Arabic text from a PDF with layout awareness and page slicing.

    Parameters
    ----------
    path:
        Path to the PDF file.
    remove_headers_footers:
        Strip running headers and page numbers.
    separate_footnotes:
        Group bottom commentary/footnotes into a distinct section per page.
    skip_pages:
        Number of pages to skip from start (0-indexed).
    take_pages:
        Number of pages to extract.
    """
    if fitz is None:
        raise ImportError(
            "PyMuPDF is required to read PDF files. "
            "Install it with: pip install PyMuPDF"
        )

    doc = fitz.open(str(path))
    total_pages = len(doc)
    start_idx = max(0, skip_pages)
    if start_idx >= total_pages and total_pages > 0:
        doc.close()
        raise ValueError(
            f"skip_pages ({skip_pages}) is beyond the end of the PDF ({total_pages} total pages)."
        )

    end_idx = min(total_pages, start_idx + take_pages) if take_pages is not None else total_pages
    page_texts: List[str] = []

    for page_idx in range(start_idx, end_idx):
        page = doc[page_idx]
        # Extract blocks: (x0, y0, x1, y1, text, block_no, block_type)
        blocks = page.get_text("blocks")
        rect = page.rect

        body_blocks, footnote_blocks = classify_page_blocks(
            blocks=blocks,
            page_width=rect.width,
            page_height=rect.height,
            remove_headers_footers=remove_headers_footers,
            separate_footnotes=separate_footnotes,
        )

        page_parts: List[str] = []
        if body_blocks:
            page_parts.append("\n\n".join(body_blocks))

        if footnote_blocks:
            footnotes_section = (
                "[الهوامش والتعليقات / Page Footnotes]:\n" + "\n".join(footnote_blocks)
            )
            page_parts.append(footnotes_section)

        if page_parts:
            page_texts.append("\n\n".join(page_parts))

    doc.close()
    return "\n\n".join(page_texts)
