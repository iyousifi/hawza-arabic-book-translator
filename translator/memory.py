"""Dynamic Session Terminology Memory.

Maintains terminology consistency across long, multi-chunk book translations
by accumulating novel scholastic terms and injecting them into subsequent chunk prompts.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from translator.normalizer import normalize_arabic_text


# Common religious honorifics to ignore when extracting terms
_HONORIFIC_PHRASES = {
    "صلى الله عليه وآله",
    "صلى الله عليه وآله وسلم",
    "عليه السلام",
    "عليهم السلام",
    "سلام الله عليه",
    "رضي الله عنه",
    "قدس سره",
    "رحمه الله",
    "عز وجل",
    "سبحانه وتعالى",
}

_LEAD_IN_WORDS = {
    "the", "a", "an", "that", "this", "these", "those",
    "is", "are", "was", "were", "be", "to", "of", "in",
    "for", "with", "by", "on", "at", "from", "as", "it",
    "and", "or", "author", "authors", "establishes", "states",
    "defines", "explains", "mentions", "calls", "called",
    "known", "namely", "e.g.", "viz.", "termed", "term",
}


def _clean_target_phrase(phrase: str) -> str:
    """Strip leading grammatical function words and punctuation from extracted phrases."""
    cleaned = phrase.strip(" '`*\"“”„«»").strip()
    words = cleaned.split()
    while words and words[0].lower() in _LEAD_IN_WORDS:
        words.pop(0)
    return " ".join(words).strip()


@dataclass
class TermRecord:
    """Record of a scholastic term tracked in session memory."""

    arabic: str
    target: str
    source_chunk: int = 0  # -1 = static glossary, 0 = preloaded session, >0 = chunk index
    confidence: float = 1.0


class TerminologyMemory:
    """Bank of dynamically accumulated scholastic terms for the translation session."""

    def __init__(self, initial_terms: Optional[Dict[str, str]] = None) -> None:
        self._terms: Dict[str, TermRecord] = {}
        if initial_terms:
            for ar, tgt in initial_terms.items():
                self.record_term(ar, tgt, source_chunk=0)

    @classmethod
    def from_file(cls, path: str | Path) -> "TerminologyMemory":
        """Instantiate TerminologyMemory loaded from a JSON file."""
        memory = cls()
        memory.load_json(path)
        return memory

    def normalize_term_key(self, arabic: str) -> str:
        """Normalize Arabic term for canonical dictionary keying."""
        cleaned = normalize_arabic_text(arabic.strip(), strip_harakat=True)
        # Collapse multiple whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def record_term(
        self,
        arabic: str,
        target: str,
        source_chunk: int = 0,
        overwrite: bool = False,
    ) -> bool:
        """Record or update an Arabic -> Target term pair.

        Returns True if a new term was added, False if skipped.
        """
        ar_clean = self.normalize_term_key(arabic)
        tgt_clean = target.strip().rstrip(";,.")

        if not ar_clean or not tgt_clean:
            return False

        # Ignore obvious honorifics
        if any(h in ar_clean for h in _HONORIFIC_PHRASES):
            return False

        # Ignore if term has too many words (likely a full sentence/clause, not a term)
        if len(ar_clean.split()) > 6 or len(tgt_clean.split()) > 8:
            return False

        if ar_clean in self._terms and not overwrite:
            return False

        self._terms[ar_clean] = TermRecord(
            arabic=ar_clean,
            target=tgt_clean,
            source_chunk=source_chunk,
        )
        return True

    def record_terms(
        self,
        mapping: Dict[str, str],
        source_chunk: int = 0,
        overwrite: bool = False,
    ) -> int:
        """Record multiple terms. Returns count of newly added terms."""
        added = 0
        for ar, tgt in mapping.items():
            if self.record_term(ar, tgt, source_chunk=source_chunk, overwrite=overwrite):
                added += 1
        return added

    def get_term(self, arabic: str) -> Optional[str]:
        """Retrieve target translation for an Arabic term if present."""
        key = self.normalize_term_key(arabic)
        rec = self._terms.get(key)
        return rec.target if rec else None

    def has_term(self, arabic: str) -> bool:
        """Check if an Arabic term is already recorded."""
        return self.normalize_term_key(arabic) in self._terms

    def novel_terms(self, include_static: bool = False) -> Dict[str, str]:
        """Return dict of {arabic: target} for novel terms learned in this session."""
        return {
            rec.arabic: rec.target
            for rec in self._terms.values()
            if include_static or rec.source_chunk >= 0
        }

    def has_novel_terms(self, include_static: bool = False) -> bool:
        """Check if any novel terms have been registered."""
        return bool(self.novel_terms(include_static=include_static))

    def format_for_prompt(self, max_terms: int = 40, include_static: bool = False) -> str:
        """Format accumulated terms for prompt injection into subsequent chunks.

        Returns empty string if no terms are available.
        """
        terms = self.novel_terms(include_static=include_static)
        if not terms:
            return ""

        # Limit to max_terms to avoid prompt bloat
        items = list(terms.items())[-max_terms:]
        lines = [f"- {ar}: {tgt}" for ar, tgt in items]

        return (
            "[Dynamic Session Terminology (Maintained from earlier sections of this book)]:\n"
            "Maintain strict consistency with these established translations:\n"
            + "\n".join(lines)
        )

    def extract_from_audit(self, audit_text: str) -> Dict[str, str]:
        """Extract novel terms from Pass 2 audit output (e.g. ### NOVEL TERMS: section)."""
        novel: Dict[str, str] = {}
        if not audit_text:
            return novel

        # Look for ### NOVEL TERMS: section if present
        section_pattern = r"###\s*NOVEL\s+TERMS:\s*([\s\S]*?)(?=###|$)"
        match = re.search(section_pattern, audit_text, flags=re.IGNORECASE)
        content = match.group(1).strip() if match else audit_text

        # Parse bullet lines like:
        # - Arabic: Target
        # * Arabic = Target
        # - Target (Arabic)
        line_pattern = re.compile(
            r"^\s*[-*•]?\s*([^\n\r:=→\-]+?)\s*[:=→\-]\s*([^\n\r]+)$", re.MULTILINE
        )
        for m in line_pattern.finditer(content):
            left = m.group(1).strip()
            right = m.group(2).strip()

            has_ar_left = bool(re.search(r"[\u0600-\u06FF]", left))
            has_ar_right = bool(re.search(r"[\u0600-\u06FF]", right))

            if has_ar_left and not has_ar_right:
                novel[left] = right
            elif has_ar_right and not has_ar_left:
                novel[right] = left

        return novel

    def extract_from_text(self, text: str) -> Dict[str, str]:
        """Heuristically extract novel bilingual terminology pairs from translated text.

        Recognizes patterns like:
        - "target term (arabic term)"
        - "arabic term (target term)"
        - "[i.e., arabic term: target term]"
        - "[i.e., target term: arabic term]"
        """
        extracted: Dict[str, str] = {}
        if not text:
            return extracted

        # Pattern 1: [i.e., Term: Translation] or [i.e., Translation: Term]
        bracket_re = re.compile(
            r"\[(?:i\.e\.,?\s*)?([^\]:\n\r]+)[:=]([^\]\n\r]+)\]",
            re.IGNORECASE,
        )
        for m in bracket_re.finditer(text):
            side1 = m.group(1).strip()
            side2 = m.group(2).strip()
            has_ar1 = bool(re.search(r"[\u0600-\u06FF]", side1))
            has_ar2 = bool(re.search(r"[\u0600-\u06FF]", side2))

            if has_ar1 and not has_ar2:
                self.record_term(side1, side2)
                extracted[side1] = side2
            elif has_ar2 and not has_ar1:
                self.record_term(side2, side1)
                extracted[side2] = side1

        # Pattern 2: Target (Arabic) or Arabic (Target) in parentheses
        paren_re = re.compile(r"([A-Za-z\s'-]{3,60})\s*\(([\u0600-\u06FF\s]{3,40})\)")
        for m in paren_re.finditer(text):
            target = _clean_target_phrase(m.group(1))
            arabic = m.group(2).strip()
            if target and self.record_term(arabic, target):
                extracted[arabic] = target

        paren_reverse_re = re.compile(r"([\u0600-\u06FF\s]{3,40})\s*\(([A-Za-z\s'-]{3,60})\)")
        for m in paren_reverse_re.finditer(text):
            arabic = m.group(1).strip()
            target = _clean_target_phrase(m.group(2))
            if target and self.record_term(arabic, target):
                extracted[arabic] = target

        return extracted

    def save_json(self, path: str | Path, include_static: bool = False) -> None:
        """Save accumulated session terms to a JSON file."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "session_terms": self.novel_terms(include_static=include_static),
        }
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_json(
        self,
        path: str | Path,
        source_chunk: int = 0,
        overwrite: bool = False,
    ) -> int:
        """Load terms from an existing JSON file.

        Supports:
        - {"session_terms": {"term": "trans"}}
        - Flat dictionary {"term": "trans"}
        - Category-grouped glossaries {"category": {"term": "trans"}}
        """
        source_path = Path(path)
        if not source_path.exists():
            raise FileNotFoundError(f"Session glossary file not found: {source_path}")

        with open(source_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        terms_to_add: Dict[str, str] = {}
        if isinstance(data, dict):
            if "session_terms" in data and isinstance(data["session_terms"], dict):
                terms_to_add.update(data["session_terms"])
            else:
                for k, v in data.items():
                    if isinstance(v, dict):
                        # Nested category
                        terms_to_add.update(v)
                    elif isinstance(v, str):
                        # Flat pair
                        terms_to_add[k] = v

        return self.record_terms(terms_to_add, source_chunk=source_chunk, overwrite=overwrite)

    def __len__(self) -> int:
        return len(self._terms)
