"""Two-Pass Scholastic Verification & Self-Critique Auditor."""

from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from rich.console import Console

from translator.config import (
    GLOSSARY_ADDENDUM_TEMPLATE,
    VERIFICATION_USER_PROMPT_TEMPLATE,
    ModelConfig,
    build_verifier_system_prompt,
    get_language_name,
)

console = Console()

MAX_RETRIES = 10
INITIAL_BACKOFF = 4


def _retry_with_backoff(fn, retries: int = MAX_RETRIES):
    """Call *fn* with exponential back-off + jitter on transient errors."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            err_str = str(exc).lower()
            if any(
                kw in err_str
                for kw in ("rate", "limit", "429", "500", "502", "503", "overloaded")
            ):
                wait = INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 2)
                console.print(
                    f"  [yellow]⏳ Rate-limited (verifier attempt {attempt + 1}/{retries}), "
                    f"retrying in {wait:.1f}s…[/yellow]"
                )
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Verifier failed after {retries} retries.")


# ---------------------------------------------------------------------------
# Data Models & Response Parsing
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    """Outcome of the scholastic audit and verification pass."""

    draft_translation: str
    verified_translation: str
    audit_notes: str
    has_corrections: bool
    novel_terms: Dict[str, str] = field(default_factory=dict)


def parse_verification_response(
    response_text: str,
    extract_terms: bool = False,
    fallback_translation: str = "",
) -> Tuple[str, str] | Tuple[str, str, Dict[str, str]]:
    """Parse verifier output into (audit_notes, verified_translation[, novel_terms])."""
    pattern = r"###\s*VERIFIED\s+TRANSLATION:\s*"
    parts = re.split(pattern, response_text, maxsplit=1, flags=re.IGNORECASE)

    audit_notes = "No structured audit header detected."
    # If no ### VERIFIED TRANSLATION: header is found, safely fall back to the Pass 1 draft
    # rather than polluting the translation with audit thoughts/critique notes.
    verified = fallback_translation if fallback_translation else response_text.strip()
    novel_terms: Dict[str, str] = {}

    if len(parts) == 2:
        audit_part = parts[0].strip()
        verified_candidate = parts[1].strip()
        if verified_candidate:
            verified = verified_candidate
        elif fallback_translation:
            verified = fallback_translation

        # Check for ### NOVEL TERMS: section
        terms_pattern = r"###\s*NOVEL\s+TERMS:\s*"
        sub_parts = re.split(terms_pattern, audit_part, maxsplit=1, flags=re.IGNORECASE)
        if len(sub_parts) == 2:
            audit_notes = re.sub(
                r"^###\s*AUDIT\s+NOTES:\s*", "", sub_parts[0], flags=re.IGNORECASE
            ).strip()
            terms_text = sub_parts[1].strip()

            line_pattern = re.compile(
                r"^\s*[-*•]?\s*([^\n\r:=→\-]+?)\s*[:=→\-]\s*([^\n\r]+)$", re.MULTILINE
            )
            for m in line_pattern.finditer(terms_text):
                left = m.group(1).strip()
                right = m.group(2).strip()
                if left.lower() in ("none", "n/a", "no novel terms"):
                    continue
                has_ar_left = bool(re.search(r"[\u0600-\u06FF]", left))
                has_ar_right = bool(re.search(r"[\u0600-\u06FF]", right))
                if has_ar_left and not has_ar_right:
                    novel_terms[left] = right
                elif has_ar_right and not has_ar_left:
                    novel_terms[right] = left
        else:
            audit_notes = re.sub(
                r"^###\s*AUDIT\s+NOTES:\s*", "", audit_part, flags=re.IGNORECASE
            ).strip()

    # Fail-safe tag preservation for bilingual mode:
    # If fallback_translation was tagged with [P1], [P2], etc., ensure that
    # any missing or truncated tags in the verified output are backfilled from the draft.
    if fallback_translation:
        from translator.bilingual import parse_tagged_paragraphs

        draft_tags = parse_tagged_paragraphs(fallback_translation)
        if draft_tags:
            verified_tags = parse_tagged_paragraphs(verified)
            missing = [
                idx for idx in draft_tags if idx not in verified_tags or not verified_tags[idx].strip()
            ]
            if missing:
                console.print(
                    f"  [dim yellow]⚠️ Verifier response missed/truncated {len(missing)} tag(s) "
                    f"({missing[:5]}{'...' if len(missing) > 5 else ''}); backfilled from Pass 1 draft.[/dim yellow]"
                )
                for idx in missing:
                    verified_tags[idx] = draft_tags[idx]

                reconstructed = [
                    f"[P{idx}]\n{verified_tags[idx]}" for idx in sorted(verified_tags.keys())
                ]
                verified = "\n\n".join(reconstructed)

    if extract_terms:
        return audit_notes, verified, novel_terms
    return audit_notes, verified


# ---------------------------------------------------------------------------
# Provider Verification Calls
# ---------------------------------------------------------------------------

def _call_gemini_verifier(
    user_msg: str,
    system_prompt: str,
    cfg: ModelConfig,
) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    def _call():
        resp = client.models.generate_content(
            model=cfg.resolved_verifier_model(),
            contents=user_msg,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=cfg.temperature,
                max_output_tokens=cfg.max_output_tokens,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        return resp.text.strip()

    return _retry_with_backoff(_call)


def _call_openai_verifier(
    user_msg: str,
    system_prompt: str,
    cfg: ModelConfig,
) -> str:
    from openai import OpenAI

    client = OpenAI()

    def _call():
        resp = client.chat.completions.create(
            model=cfg.resolved_verifier_model(),
            temperature=cfg.temperature,
            max_tokens=cfg.max_output_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        )
        return resp.choices[0].message.content.strip()

    return _retry_with_backoff(_call)


def _call_anthropic_verifier(
    user_msg: str,
    system_prompt: str,
    cfg: ModelConfig,
) -> str:
    from anthropic import Anthropic

    client = Anthropic()

    def _call():
        resp = client.messages.create(
            model=cfg.resolved_verifier_model(),
            max_tokens=cfg.max_output_tokens,
            temperature=cfg.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )
        return resp.content[0].text.strip()

    return _retry_with_backoff(_call)


# ---------------------------------------------------------------------------
# Public Verification API
# ---------------------------------------------------------------------------

def verify_chunk(
    draft_translation: str,
    arabic_chunk: str,
    cfg: ModelConfig,
    glossary_text: Optional[str] = None,
    session_terms_block: Optional[str] = None,
) -> VerificationResult:
    """Execute Pass 2: Audit and refine candidate translation against Arabic source text.

    Parameters
    ----------
    draft_translation:
        The candidate translation from Pass 1.
    arabic_chunk:
        The original classical Arabic text.
    cfg:
        Model and language configuration.
    glossary_text:
        Optional formatted glossary string.
    session_terms_block:
        Optional dynamic session terms established in earlier chunks.

    Returns
    -------
    VerificationResult
        Result containing the verified translation, audit notes, and correction flag.
    """
    lang_name = get_language_name(cfg.target_lang)

    # 1. Build system instruction for verifier
    system_prompt = build_verifier_system_prompt(
        target_lang=cfg.target_lang,
        domain=cfg.domain,
        inline_arabic=cfg.inline_arabic,
    )
    if glossary_text:
        system_prompt += GLOSSARY_ADDENDUM_TEMPLATE.format(
            target_lang_name=lang_name,
            glossary_text=glossary_text,
        )
    if session_terms_block:
        system_prompt += f"\n\n{session_terms_block.strip()}\n"

    # 2. Build audit user message
    user_msg = VERIFICATION_USER_PROMPT_TEMPLATE.format(
        target_lang_name=lang_name,
        arabic_chunk=arabic_chunk,
        candidate_translation=draft_translation,
    )

    # 3. Dispatch to verifier provider
    provider = cfg.resolved_verifier_provider()
    if provider == "anthropic":
        raw_response = _call_anthropic_verifier(user_msg, system_prompt, cfg)
    elif provider == "openai":
        raw_response = _call_openai_verifier(user_msg, system_prompt, cfg)
    else:
        raw_response = _call_gemini_verifier(user_msg, system_prompt, cfg)

    # 4. Parse audit notes, verified translation, and novel terms
    audit_notes, verified_translation, novel_terms = parse_verification_response(
        raw_response, extract_terms=True, fallback_translation=draft_translation
    )
    has_corrections = bool(verified_translation != draft_translation)

    return VerificationResult(
        draft_translation=draft_translation,
        verified_translation=verified_translation,
        audit_notes=audit_notes,
        has_corrections=has_corrections,
        novel_terms=novel_terms,
    )

