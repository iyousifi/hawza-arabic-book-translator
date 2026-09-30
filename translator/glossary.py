"""Glossary loading, formatting, and multi-language management utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

# Directory containing modular language and domain glossaries
_REPO_ROOT = Path(__file__).resolve().parent.parent
_GLOSSARIES_DIR = _REPO_ROOT / "glossaries"
_LEGACY_GLOSSARY_PATH = _REPO_ROOT / "glossary.json"


def resolve_glossary_path(
    path: Optional[str] = None,
    lang: str = "en",
    domain: str = "mantiq",
) -> Path:
    """Resolve the glossary file path based on explicit path, language, and domain."""
    if path:
        return Path(path)

    lang_code = lang.lower().strip()
    domain_code = domain.lower().strip()

    # 1. Exact match in glossaries/<lang>/<domain>.json
    candidate = _GLOSSARIES_DIR / lang_code / f"{domain_code}.json"
    if candidate.exists():
        return candidate

    # 2. Domain fallback in English: glossaries/en/<domain>.json
    candidate_en = _GLOSSARIES_DIR / "en" / f"{domain_code}.json"
    if candidate_en.exists():
        return candidate_en

    # 3. First available in specified language directory
    lang_dir = _GLOSSARIES_DIR / lang_code
    if lang_dir.exists():
        json_files = list(lang_dir.glob("*.json"))
        if json_files:
            return json_files[0]

    # 4. Fallback to legacy root glossary.json
    return _LEGACY_GLOSSARY_PATH


def load_glossary(
    path: Optional[str] = None,
    lang: str = "en",
    domain: str = "mantiq",
) -> Dict[str, Dict[str, str]]:
    """Load a glossary JSON file.

    Parameters
    ----------
    path:
        Explicit path to a JSON glossary. Overrides lang and domain if provided.
    lang:
        Target language code (e.g., 'en', 'da', 'sw').
    domain:
        Scholarly domain (e.g., 'mantiq', 'fiqh').

    Returns
    -------
    Dict[str, Dict[str, str]]
        The loaded glossary dictionary {category: {arabic_term: translation}}.
    """
    glossary_path = resolve_glossary_path(path=path, lang=lang, domain=domain)
    if not glossary_path.exists():
        return {}

    with open(glossary_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def list_available_glossaries() -> Dict[str, List[str]]:
    """Return all available language codes and their supported domains."""
    available: Dict[str, List[str]] = {}
    if not _GLOSSARIES_DIR.exists():
        return available

    for lang_dir in _GLOSSARIES_DIR.iterdir():
        if lang_dir.is_dir():
            domains = [p.stem for p in lang_dir.glob("*.json")]
            if domains:
                available[lang_dir.name] = sorted(domains)
    return available


def flatten_glossary(glossary: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """Flatten categorised glossary into a single {arabic: target_lang} mapping."""
    flat: Dict[str, str] = {}
    for _category, terms in glossary.items():
        flat.update(terms)
    return flat


def format_glossary_for_prompt(glossary: Dict[str, Dict[str, str]]) -> str:
    """Format the glossary as a readable table to inject into the system prompt."""
    lines: list[str] = []
    for category, terms in glossary.items():
        header = category.replace("_", " ").title()
        lines.append(f"\n### {header}")
        for arabic, translation in terms.items():
            lines.append(f"- {arabic} → {translation}")
    return "\n".join(lines)
