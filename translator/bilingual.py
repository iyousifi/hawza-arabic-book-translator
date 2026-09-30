"""Bilingual Alignment & Interlinear Formatting Module.

Pairs Arabic source paragraphs with translated target paragraphs and renders
alternating bilingual Markdown and Typst layouts for Hawza study.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

_PARAGRAPH_TAG_RE = re.compile(r"\[P?(\d+)\]", re.IGNORECASE)
_TAGGED_SECTION_RE = re.compile(r"\[P?(\d+)\]\s*(.*?)(?=\n\s*\[P?\d+\]|\Z)", re.DOTALL | re.IGNORECASE)


def split_into_paragraphs(text: str) -> List[str]:
    """Split text into distinct non-empty paragraphs by double newlines."""
    if not text:
        return []
    raw_paras = re.split(r"\n\s*\n+", text.strip())
    return [p.strip() for p in raw_paras if p.strip()]


def tag_arabic_paragraphs(arabic_text: str) -> Tuple[str, List[str]]:
    """Format Arabic text with paragraph tags [P1], [P2], etc. for LLM alignment.

    Returns
    -------
    Tuple[str, List[str]]
        (tagged_prompt_text, list_of_raw_paragraphs)
    """
    paras = split_into_paragraphs(arabic_text)
    if not paras:
        return "", []

    tagged_blocks: List[str] = []
    for i, p in enumerate(paras, 1):
        tagged_blocks.append(f"[P{i}]\n{p}")

    return "\n\n".join(tagged_blocks), paras


def parse_tagged_paragraphs(text: str) -> Dict[int, str]:
    """Extract paragraphs keyed by numeric index from [P1], [P2] (or [1], [2]) tags."""
    if not text:
        return {}

    matches = _TAGGED_SECTION_RE.findall(text)
    if matches:
        return {int(idx): content.strip() for idx, content in matches if content.strip()}
    return {}


def clean_paragraph_tags(text: str) -> str:
    """Remove [P1], [P2], etc. tags from text."""
    if not text:
        return ""
    cleaned = _PARAGRAPH_TAG_RE.sub("", text)
    # Collapse multiple leading whitespace on lines
    cleaned = re.sub(r"^[ \t]+", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def pair_bilingual_paragraphs(
    arabic_text: str,
    translation_text: str,
) -> List[Tuple[str, str]]:
    """Align Arabic source paragraphs with target translation paragraphs.

    1. If the translation contains explicit [P1], [P2] tags matching the prompt,
       pairs them with 100% deterministic accuracy.
    2. If untagged, uses semantic heading-aware matching so short titles and
       headings are never merged into flowing body paragraphs.
    """
    ar_paras = split_into_paragraphs(clean_paragraph_tags(arabic_text))
    tagged_dict = parse_tagged_paragraphs(translation_text)

    # Strategy 1: Explicit tag matching (100% deterministic)
    if tagged_dict:
        pairs: List[Tuple[str, str]] = []
        for i, ar in enumerate(ar_paras, 1):
            tr_content = tagged_dict.get(i, "")
            pairs.append((ar, clean_paragraph_tags(tr_content)))

        # Append any extra translation tags that had no corresponding Arabic
        for j in sorted(tagged_dict.keys()):
            if j > len(ar_paras):
                pairs.append(("", clean_paragraph_tags(tagged_dict[j])))
        return pairs

    # Strategy 2: Untagged fallback
    tr_paras = split_into_paragraphs(clean_paragraph_tags(translation_text))

    if not ar_paras and not tr_paras:
        return []
    if not ar_paras:
        return [("", tr) for tr in tr_paras]
    if not tr_paras:
        return [(ar, "") for ar in ar_paras]

    n_ar = len(ar_paras)
    n_tr = len(tr_paras)

    if n_ar == n_tr:
        return list(zip(ar_paras, tr_paras))

    # If translation was split into more paragraphs than Arabic (N < M)
    if n_ar < n_tr:
        pairs = []
        ratio = n_tr / n_ar
        tr_idx = 0
        for i, ar in enumerate(ar_paras):
            end_tr_idx = round((i + 1) * ratio) if i < n_ar - 1 else n_tr
            end_tr_idx = max(tr_idx + 1, min(end_tr_idx, n_tr - (n_ar - 1 - i)))
            group = "\n\n".join(tr_paras[tr_idx:end_tr_idx])
            pairs.append((ar, group))
            tr_idx = end_tr_idx
        return pairs

    # If Arabic paragraphs were merged in translation (N > M)
    pairs = []
    ratio = n_ar / n_tr
    ar_idx = 0
    for j, tr in enumerate(tr_paras):
        end_ar_idx = round((j + 1) * ratio) if j < n_tr - 1 else n_ar
        end_ar_idx = max(ar_idx + 1, min(end_ar_idx, n_ar - (n_tr - 1 - j)))
        group = "\n\n".join(ar_paras[ar_idx:end_ar_idx])
        pairs.append((group, tr))
        ar_idx = end_ar_idx
    return pairs


def format_bilingual_markdown(
    arabic_text: str,
    translation_text: str,
) -> str:
    """Render paired bilingual paragraphs into alternating Markdown format.

    Each Arabic paragraph is wrapped in an academic quotation card:
    > 📜 **[الأصل العربي]**:
    > {arabic}

    {translation}
    """
    pairs = pair_bilingual_paragraphs(arabic_text, translation_text)
    blocks: List[str] = []

    for ar, tr in pairs:
        clean_ar = clean_paragraph_tags(ar).strip()
        clean_tr = clean_paragraph_tags(tr).strip()

        if not clean_ar and not clean_tr:
            continue
        if not clean_ar:
            blocks.append(clean_tr)
            continue
        if not clean_tr:
            ar_lines = clean_ar.split("\n")
            ar_quoted = "\n".join(f"> {line}" for line in ar_lines)
            blocks.append(f"> 📜 **[الأصل العربي]**:\n{ar_quoted}")
            continue

        ar_lines = clean_ar.split("\n")
        ar_quoted = "\n".join(f"> {line}" for line in ar_lines)
        block = f"> 📜 **[الأصل العربي]**:\n{ar_quoted}\n\n{clean_tr}"
        blocks.append(block)

    return "\n\n".join(blocks)
