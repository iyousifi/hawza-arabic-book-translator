"""Translation engine: chunked LLM-powered classical Arabic translation with context chaining."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import random
import time
from typing import Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from translator.chunker import (
    TextChunk,
    chunk_classical_arabic,
    get_trailing_context,
)
from translator.config import (
    GLOSSARY_ADDENDUM_TEMPLATE,
    USER_PROMPT_TEMPLATE,
    USER_PROMPT_WITH_CONTEXT_TEMPLATE,
    ModelConfig,
    build_system_prompt,
    get_language_name,
)
from translator.glossary import format_glossary_for_prompt, load_glossary
from translator.memory import TerminologyMemory

import sys

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure") and sys.stdout:
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure") and sys.stderr:
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

console = Console()


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

MAX_RETRIES = 10
INITIAL_BACKOFF = 4  # seconds


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
                    f"  [yellow]⏳ Rate-limited (attempt {attempt + 1}/{retries}), "
                    f"retrying in {wait:.1f}s…[/yellow]"
                )
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {retries} retries.")


# ---------------------------------------------------------------------------
# Message formatting with context
# ---------------------------------------------------------------------------

def _build_user_message(
    chunk: str,
    target_lang: str,
    prev_target_context: Optional[str] = None,
    prev_arabic_context: Optional[str] = None,
    session_terms_block: Optional[str] = None,
) -> str:
    """Format the user prompt with target language, dual antecedent context, and dynamic session terms."""
    lang_name = get_language_name(target_lang)
    context_parts: List[str] = []

    if prev_arabic_context and prev_arabic_context.strip():
        context_parts.append(
            f"[Preceding Arabic Context (for antecedent & pronoun reference only — do NOT re-translate)]:\n"
            f'"""\n{prev_arabic_context.strip()}\n"""'
        )

    if prev_target_context and prev_target_context.strip():
        context_parts.append(
            f"[Preceding {lang_name} Translation (for terminology consistency only)]:\n"
            f'"""\n{prev_target_context.strip()}\n"""'
        )

    if session_terms_block and session_terms_block.strip():
        context_parts.append(session_terms_block.strip())

    if context_parts:
        return USER_PROMPT_WITH_CONTEXT_TEMPLATE.format(
            chunk=chunk,
            target_lang_name=lang_name,
            context_block="\n\n".join(context_parts),
        )

    return USER_PROMPT_TEMPLATE.format(
        chunk=chunk,
        target_lang_name=lang_name,
    )


# ---------------------------------------------------------------------------
# Provider-specific translation calls
# ---------------------------------------------------------------------------

def _translate_chunk_gemini(
    chunk: str,
    system_prompt: str,
    cfg: ModelConfig,
    prev_target_context: Optional[str] = None,
    prev_arabic_context: Optional[str] = None,
    session_terms_block: Optional[str] = None,
) -> str:
    """Translate a single chunk using the Google Gemini API."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    user_msg = _build_user_message(
        chunk=chunk,
        target_lang=cfg.target_lang,
        prev_target_context=prev_target_context,
        prev_arabic_context=prev_arabic_context,
        session_terms_block=session_terms_block,
    )

    def _call():
        resp = client.models.generate_content(
            model=cfg.resolved_model(),
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


def _translate_chunk_openai(
    chunk: str,
    system_prompt: str,
    cfg: ModelConfig,
    prev_target_context: Optional[str] = None,
    prev_arabic_context: Optional[str] = None,
    session_terms_block: Optional[str] = None,
) -> str:
    """Translate a single chunk using the OpenAI API."""
    from openai import OpenAI

    client = OpenAI()
    user_msg = _build_user_message(
        chunk=chunk,
        target_lang=cfg.target_lang,
        prev_target_context=prev_target_context,
        prev_arabic_context=prev_arabic_context,
        session_terms_block=session_terms_block,
    )

    def _call():
        resp = client.chat.completions.create(
            model=cfg.resolved_model(),
            temperature=cfg.temperature,
            max_tokens=cfg.max_output_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        )
        return resp.choices[0].message.content.strip()

    return _retry_with_backoff(_call)


def _translate_chunk_anthropic(
    chunk: str,
    system_prompt: str,
    cfg: ModelConfig,
    prev_target_context: Optional[str] = None,
    prev_arabic_context: Optional[str] = None,
    session_terms_block: Optional[str] = None,
) -> str:
    """Translate a single chunk using the Anthropic API."""
    from anthropic import Anthropic

    client = Anthropic()
    user_msg = _build_user_message(
        chunk=chunk,
        target_lang=cfg.target_lang,
        prev_target_context=prev_target_context,
        prev_arabic_context=prev_arabic_context,
        session_terms_block=session_terms_block,
    )

    def _call():
        resp = client.messages.create(
            model=cfg.resolved_model(),
            max_tokens=cfg.max_output_tokens,
            temperature=cfg.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )
        return resp.content[0].text.strip()

    return _retry_with_backoff(_call)


from translator.verifier import verify_chunk


# ---------------------------------------------------------------------------
# Verification Audit Reporting
# ---------------------------------------------------------------------------

@dataclass
class AuditReportEntry:
    """Record of Pass 2 audit findings for a single chunk."""

    chunk_index: int
    total_chunks: int
    has_corrections: bool
    audit_notes: str
    novel_terms: Dict[str, str] = field(default_factory=dict)


def generate_audit_report(
    entries: List[AuditReportEntry],
    target_lang: str,
    domain: str,
    provider: str,
    model: str,
    verifier_model: str,
) -> str:
    """Format an audit report in GitHub-flavored Markdown for post-run review."""
    lang_name = get_language_name(target_lang)
    corrected_count = sum(1 for e in entries if e.has_corrections)
    total = len(entries)

    lines = [
        "# 🔍 Scholastic Verification & Audit Report",
        "",
        f"- **Target Language**: {lang_name} (`{target_lang}`)",
        f"- **Domain**: {domain.upper()}",
        f"- **Primary Translation Model**: `{model}` ({provider})",
        f"- **Verification Auditor Model**: `{verifier_model}`",
        f"- **Total Chunks Audited**: {total}",
        f"- **Chunks with Corrections**: {corrected_count} / {total}",
        "",
        "---",
        "",
    ]

    for e in entries:
        status_badge = "✏️ Corrected" if e.has_corrections else "✅ Clean (No deviations detected)"
        lines.append(f"## Chunk {e.chunk_index} of {e.total_chunks} — {status_badge}")
        lines.append("")
        lines.append("### Audit Notes")
        lines.append(e.audit_notes.strip())
        lines.append("")
        if e.novel_terms:
            lines.append("### Novel Terms Discovered")
            for ar, tgt in e.novel_terms.items():
                lines.append(f"- **{ar}**: {tgt}")
            lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def translate(
    text: str,
    cfg: ModelConfig,
    glossary_path: Optional[str] = None,
    session_memory: Optional[TerminologyMemory] = None,
    save_session_glossary: Optional[str] = None,
    audit_report_path: Optional[str] = None,
    dry_run: bool = False,
) -> str:
    """Translate a classical Arabic text to the target language with discourse chunking and context chaining.

    Parameters
    ----------
    text:
        The complete Arabic source text.
    cfg:
        Model, language, domain, and provider configuration.
    glossary_path:
        Path to a custom glossary JSON. None resolves by (lang, domain).
    session_memory:
        Optional TerminologyMemory instance to accumulate terms across chunks.
    save_session_glossary:
        Optional file path to export accumulated session terms.
    audit_report_path:
        Optional file path to export a detailed verification audit report.
    dry_run:
        If True, skip API calls and return placeholder text.

    Returns
    -------
    str
        The complete translated text with chunks joined by double newlines.
    """
    lang_name = get_language_name(cfg.target_lang)

    # 1. Build scholarly system prompt for target language and domain
    system_prompt = build_system_prompt(
        target_lang=cfg.target_lang,
        domain=cfg.domain,
        inline_arabic=cfg.inline_arabic,
    )

    # 2. Load glossary matching language and domain
    glossary = load_glossary(path=glossary_path, lang=cfg.target_lang, domain=cfg.domain)
    glossary_text = None
    if glossary:
        glossary_text = format_glossary_for_prompt(glossary)
        system_prompt += GLOSSARY_ADDENDUM_TEMPLATE.format(
            target_lang_name=lang_name,
            glossary_text=glossary_text,
        )

    # 3. Initialize dynamic session terminology memory
    if session_memory is None:
        session_memory = TerminologyMemory()

    # Preload from file if configured
    session_file = cfg.session_glossary
    if session_file and os.path.exists(session_file):
        loaded_count = session_memory.load_json(session_file, source_chunk=0)
        console.print(f"[dim cyan]📚 Preloaded {loaded_count} term(s) from session glossary: {session_file}[/dim cyan]")

    # Seed with static glossary terms using source_chunk=-1 so memory knows they are established
    if glossary:
        for category_terms in glossary.values():
            if isinstance(category_terms, dict):
                for ar, tgt in category_terms.items():
                    session_memory.record_term(ar, tgt, source_chunk=-1, overwrite=False)

    # 4. Chunk source text using discourse-aware chunker with antecedent overlap
    chunks = chunk_classical_arabic(
        text,
        max_tokens=cfg.chunk_size,
        overlap_tokens=cfg.overlap_tokens,
    )
    total = len(chunks)

    verify_badge = " [bold cyan](Two-Pass Verification Enabled)[/bold cyan]" if cfg.verify else ""
    console.print(
        f"\n[bold green]📖 Source text split into {total} chunk(s) "
        f"(target: ~{cfg.chunk_size} tokens, overlap: {cfg.overlap_tokens} tokens)[/bold green]{verify_badge}"
    )
    console.print(
        f"[dim]   Language: {lang_name} ({cfg.target_lang}) | "
        f"Domain: {cfg.domain.upper()} | "
        f"Provider: {cfg.provider} ({cfg.resolved_model()})[/dim]\n"
    )

    if dry_run:
        console.print("[yellow]🔸 Dry-run mode — skipping API calls[/yellow]\n")
        verify_tag = " - VERIFIED" if cfg.verify else ""
        save_path = save_session_glossary or cfg.save_session_glossary
        if save_path:
            session_memory.save_json(save_path)
            console.print(f"[dim]Saved dry-run session glossary to {save_path}[/dim]")
        if cfg.verify and audit_report_path:
            from translator.file_io import write_output

            sample_entries = [
                AuditReportEntry(
                    chunk_index=i + 1,
                    total_chunks=total,
                    has_corrections=False,
                    audit_notes="Dry run placeholder verification audit notes.",
                )
                for i in range(total)
            ]
            dry_report = generate_audit_report(
                sample_entries,
                target_lang=cfg.target_lang,
                domain=cfg.domain,
                provider=cfg.provider,
                model=cfg.resolved_model(),
                verifier_model=cfg.resolved_verifier_model(),
            )
            write_output(dry_report, audit_report_path)
            console.print(f"[dim]Saved dry-run audit report to {audit_report_path}[/dim]")
        if cfg.inline_arabic:
            from translator.bilingual import format_bilingual_markdown

            translated_chunks = [
                format_bilingual_markdown(
                    c.text,
                    f"[DRY RUN - {lang_name.upper()} ({cfg.domain}){verify_tag}] Translated chunk {i + 1}/{total} "
                    f"({c.token_count} tokens)",
                )
                for i, c in enumerate(chunks)
            ]
        else:
            translated_chunks = [
                f"[DRY RUN - {lang_name.upper()} ({cfg.domain}){verify_tag}] Translated chunk {i + 1}/{total} "
                f"({c.token_count} tokens)\n\n"
                f"(Antecedent context: {len(c.prev_arabic_context)} chars)\n\n"
                f"(Original {len(c.text)} chars)"
                for i, c in enumerate(chunks)
            ]
        return "\n\n".join(translated_chunks)

    # 5. Pick provider function
    if cfg.provider == "anthropic":
        translate_fn = _translate_chunk_anthropic
    elif cfg.provider == "openai":
        translate_fn = _translate_chunk_openai
    else:
        translate_fn = _translate_chunk_gemini

    # 6. Translate each chunk with sliding antecedent context and optional Pass 2 verification
    translated_chunks: List[str] = []
    audit_entries: List[AuditReportEntry] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Translating to {lang_name}…", total=total)
        for i, chunk_item in enumerate(chunks):
            # Trailing Arabic context from Chunk N-1
            prev_arabic = chunk_item.prev_arabic_context

            # Trailing translated context from Chunk N-1
            prev_target = None
            if translated_chunks and cfg.overlap_tokens > 0:
                prev_target = get_trailing_context(
                    translated_chunks[-1], max_tokens=cfg.overlap_tokens
                )

            # Dynamic session terminology block from earlier chunks
            session_terms_block = (
                session_memory.format_for_prompt()
                if session_memory.has_novel_terms()
                else None
            )

            # Prepare source text (tagged with [P1], [P2] if inline_arabic is enabled)
            if cfg.inline_arabic:
                from translator.bilingual import tag_arabic_paragraphs

                chunk_to_translate, _ = tag_arabic_paragraphs(chunk_item.text)
            else:
                chunk_to_translate = chunk_item.text

            # --- Pass 1: Translation Draft ---
            progress.update(
                task,
                description=f"Chunk {i + 1}/{total} (Pass 1: Drafting)",
            )
            draft = translate_fn(
                chunk=chunk_to_translate,
                system_prompt=system_prompt,
                cfg=cfg,
                prev_target_context=prev_target,
                prev_arabic_context=prev_arabic,
                session_terms_block=session_terms_block,
            )

            # --- Pass 2: Self-Critique & Verification (Optional) ---
            if cfg.verify:
                progress.update(
                    task,
                    description=f"Chunk {i + 1}/{total} (Pass 2: Auditing)",
                )
                audit_res = verify_chunk(
                    draft_translation=draft,
                    arabic_chunk=chunk_to_translate,
                    cfg=cfg,
                    glossary_text=glossary_text,
                    session_terms_block=session_terms_block,
                )
                final_text = audit_res.verified_translation
                if audit_res.has_corrections:
                    console.print(
                        f"  [cyan]🔍 Verified chunk {i + 1}: {audit_res.audit_notes}[/cyan]"
                    )
                elif audit_res.audit_notes == "No structured audit header detected.":
                    console.print(
                        f"  [dim yellow]⚠️ Verifier omitted verified text header; preserved Pass 1 draft.[/dim yellow]"
                    )
                if audit_res.novel_terms:
                    added_from_audit = session_memory.record_terms(
                        audit_res.novel_terms, source_chunk=i + 1
                    )
                    if added_from_audit > 0:
                        console.print(
                            f"  [dim cyan]📚 Learned {added_from_audit} novel term(s) from audit in Chunk {i + 1}[/dim cyan]"
                        )
                audit_entries.append(
                    AuditReportEntry(
                        chunk_index=i + 1,
                        total_chunks=total,
                        has_corrections=audit_res.has_corrections,
                        audit_notes=audit_res.audit_notes,
                        novel_terms=audit_res.novel_terms,
                    )
                )
            else:
                final_text = draft

            # Heuristically register any bilingual terms explicitly marked in translation
            text_terms = session_memory.extract_from_text(final_text)
            if text_terms:
                session_memory.record_terms(text_terms, source_chunk=i + 1)

            if cfg.inline_arabic:
                from translator.bilingual import format_bilingual_markdown

                chunk_output = format_bilingual_markdown(chunk_item.text, final_text)
            else:
                chunk_output = final_text

            translated_chunks.append(chunk_output)
            progress.advance(task)

            # Delay between chunks to respect rate limits
            if i < total - 1 and cfg.chunk_delay > 0:
                time.sleep(cfg.chunk_delay)

    # Save session glossary if requested
    save_path = save_session_glossary or cfg.save_session_glossary
    if save_path:
        session_memory.save_json(save_path)
        console.print(f"[bold green]💾 Session glossary saved to:[/bold green] {save_path}")

    novel_count = len(session_memory.novel_terms())
    if novel_count > 0:
        console.print(
            f"[dim cyan]📚 Dynamic Terminology Memory: {novel_count} novel term(s) preserved across chunks.[/dim cyan]"
        )

    # Save audit report and display review panel if verification was enabled
    if cfg.verify and audit_entries:
        from translator.file_io import write_output

        report_markdown = generate_audit_report(
            entries=audit_entries,
            target_lang=cfg.target_lang,
            domain=cfg.domain,
            provider=cfg.provider,
            model=cfg.resolved_model(),
            verifier_model=cfg.resolved_verifier_model(),
        )

        if audit_report_path:
            write_output(report_markdown, audit_report_path)
            console.print(f"[bold green]📋 Audit report written to:[/bold green] {audit_report_path}")

        # Render terminal review panel
        corrected = [e for e in audit_entries if e.has_corrections]
        review_lines = [
            f"[dim]Audited {len(audit_entries)} chunk(s) | {len(corrected)} chunk(s) adjusted by verifier[/dim]\n"
        ]
        if corrected:
            review_lines.append("[bold yellow]Scholastic Audit Adjustments:[/bold yellow]")
            for e in corrected:
                review_lines.append(f"• [bold cyan]Chunk {e.chunk_index}:[/bold cyan] {e.audit_notes}")
                if e.novel_terms:
                    terms_preview = ", ".join(f"{ar} → {tgt}" for ar, tgt in e.novel_terms.items())
                    review_lines.append(f"  [dim]Terms: {terms_preview}[/dim]")
        else:
            review_lines.append("[bold green]✅ All chunks passed audit without critical logical deviations.[/bold green]")

        if audit_report_path:
            review_lines.append(f"\n[dim]Detailed report saved to: {audit_report_path}[/dim]")

        console.print()
        console.print(
            Panel(
                "\n".join(review_lines),
                title="🔍 [bold]Scholastic Verification Review[/bold]",
                border_style="cyan",
            )
        )

    console.print(f"\n[bold green]✅ Translation to {lang_name} complete![/bold green]\n")
    return "\n\n".join(translated_chunks)
