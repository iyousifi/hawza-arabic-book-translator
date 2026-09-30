"""Text and Page Slicing utilities: skip/take pages, page ranges, and chapter filtering."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ChapterInfo:
    """Represents a detected chapter or major structural section."""

    index: int
    title: str
    char_start: int
    char_end: int
    word_count: int

    def __str__(self) -> str:
        return f"{self.index}. {self.title} (~{self.word_count} words)"


# ---------------------------------------------------------------------------
# Page Range Parsing
# ---------------------------------------------------------------------------

def parse_page_range(
    pages_str: Optional[str] = None,
    skip: Optional[int] = None,
    take: Optional[int] = None,
) -> Tuple[int, Optional[int]]:
    """Parse page specification into (skip_pages, take_pages).

    Parameters
    ----------
    pages_str:
        Human-readable 1-indexed page range (e.g. "10-25", "10:25", "10:", ":20", "5").
    skip:
        Direct 0-indexed page skip count (e.g. skip(10)).
    take:
        Direct page take count (e.g. take(5)).

    Returns
    -------
    Tuple[int, Optional[int]]
        (skip_pages, take_pages)
    """
    if pages_str:
        p = pages_str.strip()

        # Format: "10-25" or "10:25" (inclusive range)
        m = re.match(r"^(\d+)\s*[-:]\s*(\d+)$", p)
        if m:
            start_1 = int(m.group(1))
            end_1 = int(m.group(2))
            if start_1 > end_1:
                raise ValueError(
                    f"Invalid page range '{pages_str}': start page {start_1} is greater than end page {end_1}."
                )
            skip_val = max(0, start_1 - 1)
            take_val = max(1, end_1 - start_1 + 1)
            return skip_val, take_val

        # Format: "10:" or "10+" (from page 10 to end)
        m = re.match(r"^(\d+)\s*[-:+]$", p)
        if m:
            start_1 = int(m.group(1))
            return max(0, start_1 - 1), None

        # Format: ":20" or "-20" (from start up to page 20)
        m = re.match(r"^[-:]\s*(\d+)$", p)
        if m:
            end_1 = int(m.group(1))
            return 0, max(1, end_1)

        # Format: "5" (single page)
        if p.isdigit():
            pg = int(p)
            return max(0, pg - 1), 1

        raise ValueError(
            f"Invalid page range format: '{pages_str}'. "
            "Supported formats: '10-25' (pages 10 to 25), '10:' (page 10 to end), "
            "':20' (first 20 pages), or '5' (page 5 only)."
        )

    skip_val = max(0, skip) if skip is not None else 0
    take_val = max(1, take) if take is not None else None
    return skip_val, take_val


def format_page_tag(skip_pages: int, take_pages: Optional[int]) -> str:
    """Format (skip_pages, take_pages) into a filename tag (e.g. 'p10-25' or 'p10-end')."""
    start_1 = skip_pages + 1
    if take_pages is not None:
        end_1 = skip_pages + take_pages
        if take_pages == 1:
            return f"p{start_1}"
        return f"p{start_1}-{end_1}"
    if skip_pages > 0:
        return f"p{start_1}-end"
    return ""


# ---------------------------------------------------------------------------
# Chapter Detection & Extraction
# ---------------------------------------------------------------------------

_HEADING_PATTERN = re.compile(
    r"(?:^|\n)([ \t]*(?:المقصد|الباب|الفصل|المبحث|المسألة|المدخل|المقدمة|الخاتمة)"
    r"(?:\s+(?:الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر|\d+|[٠-٩]+))?"
    r"[^\n]*)"
)


def detect_chapters(text: str) -> List[ChapterInfo]:
    """Scan Arabic text and detect major chapter and section boundaries."""
    if not text:
        return []

    matches = list(_HEADING_PATTERN.finditer(text))
    chapters: List[ChapterInfo] = []

    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.start(1)
        end = matches[i + 1].start(1) if i + 1 < len(matches) else len(text)
        content = text[start:end]
        word_count = len(content.split())

        chapters.append(
            ChapterInfo(
                index=i + 1,
                title=title,
                char_start=start,
                char_end=end,
                word_count=word_count,
            )
        )

    return chapters


def extract_chapter(text: str, query: str | int) -> Tuple[str, ChapterInfo]:
    """Extract the text corresponding to a specific chapter by index or title query.

    Parameters
    ----------
    text:
        The complete Arabic source text.
    query:
        Chapter number (e.g. 1, 2, "1") or title search substring (e.g. "مباحث الألفاظ").

    Returns
    -------
    Tuple[str, ChapterInfo]
        (chapter_text, chapter_info)
    """
    chapters = detect_chapters(text)
    if not chapters:
        raise ValueError(
            "No chapters or section headings were detected in the source text."
        )

    q_str = str(query).strip()

    # 1. Match by numeric index (1-based)
    if q_str.isdigit():
        target_idx = int(q_str)
        for c in chapters:
            if c.index == target_idx:
                return text[c.char_start : c.char_end].strip(), c

        available = "\n".join(f"  {c.index}. {c.title} (~{c.word_count} words)" for c in chapters)
        raise ValueError(
            f"Chapter index {target_idx} is out of range (available: 1 to {len(chapters)}).\n"
            f"Available chapters:\n{available}"
        )

    # 2. Match by title substring (case-insensitive)
    q_norm = q_str.lower()
    matches = [c for c in chapters if q_norm in c.title.lower()]

    if len(matches) == 1:
        c = matches[0]
        return text[c.char_start : c.char_end].strip(), c

    if len(matches) > 1:
        matched_list = "\n".join(f"  {c.index}. {c.title}" for c in matches)
        raise ValueError(
            f"Multiple chapters matched query '{query}':\n{matched_list}\n"
            f"Please specify by exact number (e.g. --chapter {matches[0].index})."
        )

    available = "\n".join(f"  {c.index}. {c.title}" for c in chapters)
    raise ValueError(
        f"Chapter '{query}' not found.\nAvailable chapters:\n{available}"
    )


# ---------------------------------------------------------------------------
# Text-Based Page Slicing Fallback (for .txt and .docx)
# ---------------------------------------------------------------------------

def slice_text_by_pages(
    text: str,
    skip_pages: int = 0,
    take_pages: Optional[int] = None,
    words_per_page: int = 350,
) -> str:
    """Slice non-PDF plain text or docx by simulated book pages (~350 words/page) or form-feeds."""
    if not text:
        return ""

    # If document has explicit form feed page breaks (\f)
    if "\f" in text:
        pages = text.split("\f")
    else:
        # Approximate pages by words
        words = text.split()
        total_words = len(words)
        pages = []
        for i in range(0, total_words, words_per_page):
            pages.append(" ".join(words[i : i + words_per_page]))

    total_pages = len(pages)
    start = max(0, skip_pages)
    if start >= total_pages:
        raise ValueError(
            f"skip_pages ({skip_pages}) exceeds total pages in document ({total_pages})."
        )

    end = min(total_pages, start + take_pages) if take_pages is not None else total_pages
    return "\n\n".join(pages[start:end])
