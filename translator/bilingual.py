"""Bilingual Alignment & Interlinear Formatting Module.

Pairs Arabic source paragraphs with translated target paragraphs and renders
alternating bilingual Markdown and Typst layouts for Hawza study.
"""

from __future__ import annotations

import re
from typing import List, Tuple


def split_into_paragraphs(text: str) -> List[str]:
    """Split text into distinct non-empty paragraphs by double newlines."""
    if not text:
        return []
    raw_paras = re.split(r"\n\s*\n+", text.strip())
    return [p.strip() for p in raw_paras if p.strip()]


def pair_bilingual_paragraphs(
    arabic_text: str,
    translation_text: str,
) -> List[Tuple[str, str]]:
    """Align Arabic source paragraphs with target translation paragraphs.

    Handles equal paragraph counts (1:1) and provides intelligent grouping
    when the model merges or splits paragraphs (N != M), ensuring no text is lost.
    """
    ar_paras = split_into_paragraphs(arabic_text)
    tr_paras = split_into_paragraphs(translation_text)

    if not ar_paras and not tr_paras:
        return []
    if not ar_paras:
        return [("", tr) for tr in tr_paras]
    if not tr_paras:
        return [(ar, "") for ar in ar_paras]

    n_ar = len(ar_paras)
    n_tr = len(tr_paras)

    # 1. Exact match (Ideal)
    if n_ar == n_tr:
        return list(zip(ar_paras, tr_paras))

    # 2. More translation paragraphs than Arabic (model split long sentences into multiple paragraphs)
    if n_ar < n_tr:
        pairs: List[Tuple[str, str]] = []
        ratio = n_tr / n_ar
        tr_idx = 0
        for i, ar in enumerate(ar_paras):
            end_tr_idx = round((i + 1) * ratio) if i < n_ar - 1 else n_tr
            end_tr_idx = max(tr_idx + 1, min(end_tr_idx, n_tr - (n_ar - 1 - i)))
            group = "\n\n".join(tr_paras[tr_idx:end_tr_idx])
            pairs.append((ar, group))
            tr_idx = end_tr_idx
        return pairs

    # 3. More Arabic paragraphs than translation (model merged short paragraphs)
    pairs: List[Tuple[str, str]] = []
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
        if not ar.strip() and not tr.strip():
            continue
        if not ar.strip():
            blocks.append(tr.strip())
            continue
        if not tr.strip():
            ar_lines = ar.strip().split("\n")
            ar_quoted = "\n".join(f"> {line}" for line in ar_lines)
            blocks.append(f"> 📜 **[الأصل العربي]**:\n{ar_quoted}")
            continue

        ar_lines = ar.strip().split("\n")
        ar_quoted = "\n".join(f"> {line}" for line in ar_lines)
        block = f"> 📜 **[الأصل العربي]**:\n{ar_quoted}\n\n{tr.strip()}"
        blocks.append(block)

    return "\n\n".join(blocks)
