"""Classical Discourse-Aware Chunker with Sliding Context Overlap."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

try:
    import tiktoken

    _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_ENCODER = None


# ---------------------------------------------------------------------------
# Token Counting & Windowing Helpers
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """Return the token count for text using cl100k_base or a calibrated Arabic ratio."""
    if not text:
        return 0
    if _TIKTOKEN_ENCODER is not None:
        try:
            return len(_TIKTOKEN_ENCODER.encode(text))
        except Exception:
            pass
    # Calibrated fallback for Arabic text: ~1.35 characters per BPE token
    return max(1, int(len(text) / 1.35))


def get_trailing_context(text: str, max_tokens: int = 200) -> str:
    """Extract the trailing *max_tokens* from text, trimmed to a word boundary."""
    if not text or max_tokens <= 0:
        return ""

    if _TIKTOKEN_ENCODER is not None:
        try:
            tokens = _TIKTOKEN_ENCODER.encode(text)
            if len(tokens) <= max_tokens:
                return text.strip()
            tail_tokens = tokens[-max_tokens:]
            decoded = _TIKTOKEN_ENCODER.decode(tail_tokens)
            # Trim leading partial word
            if " " in decoded:
                decoded = decoded[decoded.index(" ") + 1 :]
            return decoded.strip()
        except Exception:
            pass

    # Character-based fallback
    max_chars = int(max_tokens * 1.35)
    if len(text) <= max_chars:
        return text.strip()
    tail = text[-max_chars:]
    if " " in tail:
        tail = tail[tail.index(" ") + 1 :]
    return tail.strip()


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class TextChunk:
    """Represents a discrete translation chunk with antecedent context."""

    index: int
    text: str
    token_count: int
    prev_arabic_context: str = ""

    def __str__(self) -> str:
        return self.text


# ---------------------------------------------------------------------------
# Hierarchical Discourse Splitters
# ---------------------------------------------------------------------------

# Tier 0: Major section divisions in Hawza literature
_TIER0_HEADINGS_RE = re.compile(
    r"(?:^|\n)(?=(?:المقصد|الباب|الفصل|المبحث|المسألة|المدخل|المقدمة|الخاتمة)\s+"
    r"(?:الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر|\d+|[٠-٩]+))"
)

# Tier 1: Paragraph breaks
_TIER1_PARAGRAPH_RE = re.compile(r"\n{2,}")

# Tier 2: Dialectical objections, responses, and scholastic caveats
_TIER2_DIALECTIC_RE = re.compile(
    r"(?<=\s)(?=(?:فإن\s+قيل|إن\s+قلت|وقد\s+يقال|لا\s+يقال|لو\s+قيل|قلنا|فنقول|أجيب|والجواب|قلت|"
    r"تنبيه|تذنيب|فائدة|إيقاظ|تبصرة|إضاءة)[:\s])"
)

# Tier 3: Syllogistic premises, inferential transitions, and logical deduction markers
_TIER3_LOGIC_RE = re.compile(
    r"(?<=\s)(?=(?:وحيث\s+إن|ولما\s+كان|إذ\s+المفروض|بما\s+أن|المقدمة\s+الأولى|المقدمة\s+الثانية|"
    r"فينتج|فيلزم|إذن|ومن\s+هنا|وحينئذ|والحاصل|والمراد\s+به|والخلاصة|إذا\s+عرفت\s+هذا|ثم\s+إن|"
    r"وعلى\s+هذا|هذا\s+و|أما\s+بعد)\b)"
)

# Tier 4: Classical Arabic sentence termination punctuation
_TIER4_PUNCT_RE = re.compile(r"(?<=[.؟!؛])\s+")

# Tier 5: Clause delimiters (Arabic comma, Latin comma)
_TIER5_CLAUSE_RE = re.compile(r"(?<=[،,])\s+")

# Tier 6: Whitespace token boundary
_TIER6_SPACE_RE = re.compile(r"\s+")


def _split_recursively(text: str, max_tokens: int, tier: int = 0) -> List[str]:
    """Recursively decompose Arabic text along classical discourse boundaries."""
    text = text.strip()
    if not text:
        return []

    if count_tokens(text) <= max_tokens:
        return [text]

    # Select regex pattern for current hierarchy tier
    if tier == 0:
        parts = [p.strip() for p in _TIER0_HEADINGS_RE.split(text) if p.strip()]
    elif tier == 1:
        parts = [p.strip() for p in _TIER1_PARAGRAPH_RE.split(text) if p.strip()]
    elif tier == 2:
        parts = [p.strip() for p in _TIER2_DIALECTIC_RE.split(text) if p.strip()]
    elif tier == 3:
        parts = [p.strip() for p in _TIER3_LOGIC_RE.split(text) if p.strip()]
    elif tier == 4:
        parts = [p.strip() for p in _TIER4_PUNCT_RE.split(text) if p.strip()]
    elif tier == 5:
        parts = [p.strip() for p in _TIER5_CLAUSE_RE.split(text) if p.strip()]
    else:
        # Fallback: slice by words
        words = text.split()
        chunks: List[str] = []
        cur: List[str] = []
        for w in words:
            cur.append(w)
            if count_tokens(" ".join(cur)) >= max_tokens:
                chunks.append(" ".join(cur))
                cur = []
        if cur:
            chunks.append(" ".join(cur))
        return chunks

    # If this tier did not split the text into smaller pieces, advance tier
    if len(parts) <= 1:
        return _split_recursively(text, max_tokens, tier=tier + 1)

    # Subdivide any oversized pieces further
    result: List[str] = []
    for part in parts:
        if count_tokens(part) > max_tokens:
            result.extend(_split_recursively(part, max_tokens, tier=tier + 1))
        else:
            result.append(part)

    return result


# ---------------------------------------------------------------------------
# Public Chunker API
# ---------------------------------------------------------------------------

def chunk_classical_arabic(
    text: str,
    max_tokens: int = 1500,
    overlap_tokens: int = 200,
) -> List[TextChunk]:
    """Chunk classical Arabic text using scholastic discourse hierarchy with antecedent overlap.

    Parameters
    ----------
    text:
        Clean, normalized Arabic source text.
    max_tokens:
        Target maximum token budget per chunk.
    overlap_tokens:
        Tokens of preceding Arabic text to capture as antecedent context for each chunk.

    Returns
    -------
    List[TextChunk]
        List of TextChunk objects containing chunk text, token count, and previous Arabic context.
    """
    if not text or not text.strip():
        return []

    # 1. Decompose text into discrete scholastic units
    units = _split_recursively(text.strip(), max_tokens=max_tokens, tier=0)

    # 2. Pack units into chunks up to max_tokens
    chunks_text: List[str] = []
    current_units: List[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = count_tokens(unit)

        if current_tokens + unit_tokens > max_tokens and current_units:
            chunks_text.append("\n\n".join(current_units))
            current_units = [unit]
            current_tokens = unit_tokens
        else:
            current_units.append(unit)
            current_tokens += unit_tokens

    if current_units:
        chunks_text.append("\n\n".join(current_units))

    # 3. Build TextChunk instances with antecedent context
    result: List[TextChunk] = []
    for i, c_text in enumerate(chunks_text):
        prev_context = ""
        if i > 0 and overlap_tokens > 0:
            prev_context = get_trailing_context(chunks_text[i - 1], max_tokens=overlap_tokens)

        result.append(
            TextChunk(
                index=i,
                text=c_text,
                token_count=count_tokens(c_text),
                prev_arabic_context=prev_context,
            )
        )

    return result


def chunk_text(text: str, max_tokens: int = 1500) -> List[str]:
    """Backward-compatible helper returning raw string chunks."""
    return [c.text for c in chunk_classical_arabic(text, max_tokens=max_tokens)]
