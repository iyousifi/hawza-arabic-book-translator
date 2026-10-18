"""Bilingual Alignment & Interlinear Formatting Module.

Pairs Arabic source paragraphs with translated target paragraphs and renders
alternating bilingual Markdown and Typst layouts for Hawza study.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

from translator.normalizer import strip_arabic_vowels

logger = logging.getLogger(__name__)

_STRICT_TAG_RE = re.compile(
    r"(?:^|\n)\s*(?:[#*_\-]+\s*)?"          # optional leading markdown (###, **, -)
    r"(?:\[\s*P?\s*(\d+)\s*\]"             # [P1] or [1] or [P 1]
    r"|P\s*(\d+)\s*[:\-–—.)])"             # or P1: or P1. or P 1:
    r"[:\-–—.)]*"                           # optional trailing colon/period
    r"(?:\s*[*_]+)?\s*",                   # optional trailing bold/italic
    re.IGNORECASE,
)


def split_into_paragraphs(text: str) -> List[str]:
    """Split text into distinct non-empty paragraphs by double newlines.

    Attaches any standalone <!-- rubric: ... --> comments as a prefix to the
    subsequent paragraph so they don't count as isolated translation blocks.
    """
    if not text:
        return []
    raw_paras = re.split(r"\n\s*\n+", text.strip())
    cleaned_paras: List[str] = []
    pending_prefix = ""
    for p in raw_paras:
        clean = p.strip()
        if not clean:
            continue
        if re.match(r"^<!--\s*rubric:\s*.*?\s*-->$", clean, re.IGNORECASE):
            pending_prefix = clean + "\n"
            continue
        if pending_prefix:
            clean = pending_prefix + clean
            pending_prefix = ""
        cleaned_paras.append(clean)
    if pending_prefix and cleaned_paras:
        cleaned_paras[-1] = pending_prefix + cleaned_paras[-1]
    return cleaned_paras


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

    matches = list(_STRICT_TAG_RE.finditer(text))
    if not matches:
        return {}

    result: Dict[int, str] = {}
    for i, m in enumerate(matches):
        idx_str = m.group(1) or m.group(2)
        if not idx_str:
            continue
        idx = int(idx_str)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            result[idx] = content
    return result


def clean_paragraph_tags(text: str) -> str:
    """Remove [P1], [P2], etc. tags from text."""
    if not text:
        return ""
    cleaned = _STRICT_TAG_RE.sub("\n", text)
    # Also clean any inline [P1] or [1] remnants
    cleaned = re.sub(r"\[\s*P?\s*\d+\s*\]", "", cleaned, flags=re.IGNORECASE)
    # Collapse multiple leading whitespace on lines
    cleaned = re.sub(r"^[ \t]+", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def sanitize_translation_artifacts(text: str) -> str:
    """Strip LLM internal deliberations, conversational artifacts, and BiDi bullet hazards.

    1. Removes lines where the LLM talks to itself (e.g. 'Wait...', 'Let's see...', 'In , ...').
    2. Untangles BiDi bullet hazard lines (e.g. '- "Arabic" -> "Target"') into standard bilingual blocks.
    3. Strips lone dummy placeholders (e.g. '.', '•', lone brackets).
    """
    if not text:
        return ""

    meta_patterns = [
        r"^\s*(?:Wait[!,]|Wait\s+a\s+moment|Let'?s\s+see|Look\s+at\s+what|In\s*,\s*|Can\s+we\s+present|First,\s+the\s+completion\s+of|Notice\s+that|Here\s+is\s+the|Untangling\s+the\s+two\s+fragments).*$",
        r"^\s*If\s+we\s+read\s+.*\s+clearly!?\s*$",
        r"^\s*Wait,\s+look\s+at\s+how\s+beautiful\s+that\s+is:?\s*$",
        r"^\s*If\s+we\s+translate\s+both\s+clauses\s+in\s*,.*$",
        r"^\s*Let'?s\s+check\s*:?\s*$",
        r"^\s*(?:Right|Left)\s+text:\s*\"[^\"]*\".*$",
        r"^\s*Formatting\s+check:?\s*$",
        r"^\s*Only\s+translate\s+what\s+is\s+in.*$",
        r"^\s*Each\s+paragraph\s+tag.*must\s+be\s+strictly\s+preserved.*$",
    ]
    lines = text.splitlines()
    clean_lines = []
    for line in lines:
        if any(re.match(pat, line, re.IGNORECASE) for pat in meta_patterns):
            continue
        clean_lines.append(line)
    text = "\n".join(clean_lines)

    # Untangle bulleted Arabic -> Translation BiDi hazard lines
    def repl_bullet_arrow(m):
        ar_part = m.group(1).strip().strip("\"'«»")
        tr_part = m.group(2).strip().strip("\"'«»")
        return f"> {ar_part}\n\n{tr_part}"

    bullet_arrow_re = re.compile(
        r"^\s*[-*•]\s*[\"\'«]?([\u0600-\u06FF\s\d.,;?!:()]+)[\"\'»]?\s*(?:->|→|:)\s*[\"\'«]?(.*?)[\"\'»]?\s*$",
        re.MULTILINE,
    )
    text = bullet_arrow_re.sub(repl_bullet_arrow, text)

    return text.strip()


def get_missing_tags(arabic_text: str, translation_text: str) -> List[int]:
    """Identify 1-based indices of Arabic paragraphs that have missing or empty translation tags."""
    ar_paras = split_into_paragraphs(clean_paragraph_tags(arabic_text))
    tagged_dict = parse_tagged_paragraphs(translation_text)
    if not tagged_dict:
        return []
    missing = [i for i in range(1, len(ar_paras) + 1) if i not in tagged_dict or not tagged_dict[i].strip()]
    return missing


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
    translation_text = sanitize_translation_artifacts(translation_text)
    ar_paras = split_into_paragraphs(clean_paragraph_tags(arabic_text))
    tagged_dict = parse_tagged_paragraphs(translation_text)

    # Strategy 1: Explicit tag matching (100% deterministic)
    if tagged_dict:
        pairs: List[Tuple[str, str]] = []
        for i, ar in enumerate(ar_paras, 1):
            tr_content = tagged_dict.get(i, "")
            if not tr_content.strip() and ar.strip():
                logger.warning("Bilingual pairing: Arabic paragraph [P%d] has no translation content.", i)
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
    strip_harakat: bool = True,
    keep_shaddah: bool = False,
) -> str:
    """Render paired bilingual paragraphs into alternating Markdown format.

    Each Arabic paragraph is wrapped in an academic quotation card:
    > 📜 **[الأصل العربي]**:
    > {arabic}

    {translation}
    """
    if strip_harakat and arabic_text:
        arabic_text = strip_arabic_vowels(arabic_text, keep_shaddah=keep_shaddah)

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
            blocks.append(ar_quoted)
            continue

        ar_lines = clean_ar.split("\n")
        ar_quoted = "\n".join(f"> {line}" for line in ar_lines)
        block = f"{ar_quoted}\n\n{clean_tr}"
        blocks.append(block)

    raw_md = "\n\n".join(blocks)
    return consolidate_bilingual_markdown(raw_md, min_chars=250)


_SECTION_TITLE_RE = re.compile(
    r"^(?:الدرس|المبحث|المطلب|المقصد|الباب|الفصل|المرحلة|المسألة|أولاً|ثانياً|ثالثاً|رابعاً|خامساً)\s*(?:الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر|الحادي|الثاني|الثالث|\d+)?",
    re.IGNORECASE,
)
_FOOTNOTE_HEADER_RE = re.compile(r"\[الهوامش والتعليقات\s*/\s*Page Footnotes\]", re.IGNORECASE)


def consolidate_bilingual_markdown(md_text: str, min_chars: int = 250) -> str:
    """Consolidate fragmented bilingual markdown blocks into cohesive paragraphs of at least 3-4 lines.

    Merges contiguous small Arabic blocks and their matching translations until each paragraph
    reaches at least min_chars (~250-450 characters / 3-5 lines).
    Eliminates dummy translation placeholders (e.g. '.', '•', '[P', lone brackets) and
    prevents disjointed one-line blocks in the final document and typeset PDF.
    """
    if not md_text:
        return ""

    raw_blocks = md_text.split("\n\n")
    parsed_items: List[Tuple[str, str, str]] = []
    i = 0

    def is_boundary(ar_text: str) -> bool:
        clean = ar_text.strip()
        if clean.startswith("#"):
            return True
        if _FOOTNOTE_HEADER_RE.search(clean):
            return True
        if len(clean) < 60 and bool(_SECTION_TITLE_RE.match(clean)):
            return True
        return False

    while i < len(raw_blocks):
        b = raw_blocks[i].strip()
        if not b:
            i += 1
            continue
        if b.startswith("<!-- slice:"):
            parsed_items.append(("slice", b, ""))
            i += 1
        elif b.startswith("#"):
            parsed_items.append(("heading", b, ""))
            i += 1
        elif _FOOTNOTE_HEADER_RE.search(b):
            parsed_items.append(("footnote_header", b, ""))
            i += 1
        elif b.startswith("[^") and "]:" in b:
            parsed_items.append(("footnote_def", b, ""))
            i += 1
        elif b.startswith(">"):
            ar_text = "\n".join(l.lstrip("> ").strip() for l in b.splitlines()).strip()
            tr_text = ""
            if i + 1 < len(raw_blocks):
                next_b = raw_blocks[i + 1].strip()
                if not next_b.startswith(">") and not next_b.startswith("#") and not next_b.startswith("<!--") and not (next_b.startswith("[^") and "]:" in next_b):
                    tr_text = next_b
                    i += 1
            parsed_items.append(("bilingual", ar_text, tr_text))
            i += 1
        else:
            parsed_items.append(("text", b, ""))
            i += 1

    merged_items: List[Tuple[str, str, str]] = []
    acc_ar: List[str] = []
    acc_tr: List[str] = []

    def flush_acc():
        nonlocal acc_ar, acc_tr
        if not acc_ar:
            return
        combined_ar = " ".join(acc_ar).strip()
        clean_tr_parts = []
        for t in acc_tr:
            t_clean = t.strip()
            if t_clean in (".", "•", "-", "..", "...", "[P", "[", "]"):
                continue
            if t_clean:
                clean_tr_parts.append(t_clean)
        combined_tr = " ".join(clean_tr_parts).strip()
        merged_items.append(("bilingual", combined_ar, combined_tr))
        acc_ar = []
        acc_tr = []

    for item_type, ar, tr in parsed_items:
        if item_type != "bilingual":
            flush_acc()
            merged_items.append((item_type, ar, tr))
            continue

        if is_boundary(ar):
            flush_acc()
            merged_items.append(("bilingual", ar, tr))
            continue

        if ar in ("•", "-") and tr in ("•", "-", ""):
            continue

        acc_ar.append(ar)
        if tr and tr not in (".", "•", "-", "..", "...", "[P", "[", "]"):
            acc_tr.append(tr)

        curr_ar_len = sum(len(x) for x in acc_ar)
        last_ar = acc_ar[-1].strip()
        ends_sentence = bool(re.search(r"[.!:؟]\s*$", last_ar))
        has_substantive_tr = bool(acc_tr)

        if curr_ar_len >= min_chars and ends_sentence and has_substantive_tr:
            flush_acc()
        elif curr_ar_len >= int(min_chars * 1.9) and has_substantive_tr:
            flush_acc()

    flush_acc()

    out_blocks: List[str] = []
    for item_type, ar, tr in merged_items:
        if item_type in ("slice", "heading", "footnote_header", "footnote_def", "text"):
            out_blocks.append(ar)
        elif item_type == "bilingual":
            ar_quoted = "\n".join(f"> {l}" for l in ar.split("\n"))
            if tr:
                out_blocks.append(f"{ar_quoted}\n\n{tr}")
            else:
                out_blocks.append(ar_quoted)

    return "\n\n".join(out_blocks) + "\n"


def audit_bilingual_markdown(md_text: str) -> List[Dict[str, Any]]:
    """Audit bilingual markdown document for formatting, missing translations, and BiDi hazards.

    Returns a list of issue dictionaries containing severity, issue type, line number, and description:
    - orphaned_arabic_block: consecutive Arabic blocks without intervening translation.
    - dummy_translation_block: translations that are purely '.', '•', '-', '[P', or dummy markers.
    - bidi_bullet_hazard: raw Arabic text inside LTR markdown list bullets.
    - llm_monologue_leak: internal conversational or drafting thoughts from the LLM.
    """
    issues = []
    if not md_text:
        return issues

    raw_blocks = md_text.split("\n\n")
    prev_was_arabic = False
    prev_arabic_line = 0

    line_counter = 1
    for block in raw_blocks:
        b = block.strip()
        lines_in_block = len(block.splitlines())
        block_line = line_counter

        if not b:
            line_counter += max(1, lines_in_block) + 1
            continue

        if b.startswith("<!--") or b.startswith("#") or b.startswith("[^"):
            prev_was_arabic = False
            line_counter += lines_in_block + 1
            continue

        if _FOOTNOTE_HEADER_RE.search(b):
            prev_was_arabic = False
            line_counter += lines_in_block + 1
            continue

        # If it's a section title inside an Arabic quote (e.g. > المرحلة الثانية or > الدرس العاشر)
        clean_no_quote = "\n".join(l.lstrip("> ").strip() for l in b.splitlines()).strip()
        if len(clean_no_quote) < 60 and bool(_SECTION_TITLE_RE.match(clean_no_quote)):
            prev_was_arabic = False
            line_counter += lines_in_block + 1
            continue

        if b.startswith(">"):
            if prev_was_arabic:
                issues.append({
                    "severity": "ERROR",
                    "type": "orphaned_arabic_block",
                    "line": prev_arabic_line,
                    "description": f"Consecutive Arabic blocks without intervening translation near line {prev_arabic_line}.",
                })
            prev_was_arabic = True
            prev_arabic_line = block_line
        else:
            prev_was_arabic = False
            if b in (".", "•", "-", "..", "...", "[P", "[", "]"):
                issues.append({
                    "severity": "WARNING",
                    "type": "dummy_translation_block",
                    "line": block_line,
                    "description": f"Dummy translation placeholder '{b}' near line {block_line}.",
                })
            for l_idx, line in enumerate(b.splitlines(), start=block_line):
                l_str = line.strip()
                if l_str.startswith(("- ", "* ", "• ")) and re.search(r"[\u0600-\u06FF]{4,}", l_str):
                    issues.append({
                        "severity": "ERROR",
                        "type": "bidi_bullet_hazard",
                        "line": l_idx,
                        "description": f"Raw Arabic found inside LTR list bullet near line {l_idx}: '{l_str[:50]}...'",
                    })
                if re.match(r"^\s*(?:Wait[!,]|Let'?s\s+see|Look\s+at\s+what|In\s*,\s*|Can\s+we\s+present)\b", l_str, re.IGNORECASE):
                    issues.append({
                        "severity": "WARNING",
                        "type": "llm_monologue_leak",
                        "line": l_idx,
                        "description": f"Potential LLM internal monologue near line {l_idx}: '{l_str[:50]}...'",
                    })

        line_counter += lines_in_block + 1

    return issues
