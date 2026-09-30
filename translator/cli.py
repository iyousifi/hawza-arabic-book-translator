"""CLI entry point for the Arabic Book Translator."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from rich.console import Console
from rich.panel import Panel

from translator import __version__
from translator.config import ModelConfig, get_language_name
from translator.engine import translate
from translator.file_io import default_audit_path, default_output_path, read_file, write_output
from translator.glossary import list_available_glossaries
from translator.slicing import (
    detect_chapters,
    extract_chapter,
    format_page_tag,
    parse_page_range,
)

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




def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arabic-translator",
        description=(
            "Translate Classical and Scholastic Arabic books to English, Danish, Swahili, etc., "
            "optimised for formal Islamic scholarly works (Hawza Logic / Mantiq, Fiqh, and Falsafa)."
        ),
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to the input file (.txt, .pdf, or .docx)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path for the translated output file. Defaults to <input>_translated_<lang>.md",
    )
    parser.add_argument(
        "-l",
        "--lang",
        default="en",
        help="Target language code (e.g., 'en' for English, 'da' for Danish, 'sw' for Swahili; default: 'en')",
    )
    parser.add_argument(
        "-d",
        "--domain",
        choices=["mantiq", "fiqh", "aqida", "akhlaaq", "tadabbur", "thaqafa", "tarikh", "general"],
        default="mantiq",
        help="Scholarly domain preset (mantiq, fiqh, aqida, akhlaaq, tadabbur, thaqafa, tarikh, general; default: mantiq)",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai", "anthropic"],
        default="gemini",
        help="LLM provider to use (default: gemini)",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model name override (default depends on provider)",
    )
    parser.add_argument(
        "--glossary",
        default=None,
        help="Path to a custom glossary JSON file (overrides built-in language/domain glossary)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1500,
        help="Target tokens per chunk (default: 1500)",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=200,
        help="Antecedent context tokens passed from previous chunk (default: 200)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM sampling temperature (default: 0.2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip API calls; test file reading, normalization, and chunking only",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=4.0,
        help="Seconds to wait between API calls to avoid rate-limiting (default: 4.0, set to 0 to disable)",
    )
    parser.add_argument(
        "--keep-headers",
        action="store_true",
        help="Do not filter out running headers and footers from PDF pages",
    )
    parser.add_argument(
        "--merge-footnotes",
        action="store_true",
        help="Do not separate bottom footnotes/commentary into a dedicated section",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Enable Two-Pass Scholastic Verification to audit draft translations and eliminate smooth hallucinations",
    )
    parser.add_argument(
        "--verifier-model",
        default="",
        help="Model override for Pass 2 verification (defaults to translation model)",
    )
    parser.add_argument(
        "--verifier-provider",
        choices=["gemini", "openai", "anthropic"],
        default="",
        help="Provider override for Pass 2 verification (defaults to primary provider)",
    )
    parser.add_argument(
        "--pages",
        default=None,
        help="Page range to translate (e.g. '10-25', '10:', ':20', or '5')",
    )
    parser.add_argument(
        "--skip-pages",
        type=int,
        default=None,
        help="Number of pages to skip from start (e.g. 10)",
    )
    parser.add_argument(
        "--take-pages",
        type=int,
        default=None,
        help="Number of pages to translate (e.g. 5)",
    )
    parser.add_argument(
        "--chapter",
        default=None,
        help="Translate only a specific chapter (by number e.g. '1' or title query e.g. 'مباحث الألفاظ')",
    )
    parser.add_argument(
        "--list-chapters",
        action="store_true",
        help="Scan document and list all detected chapters and sections",
    )
    parser.add_argument(
        "--session-glossary",
        default=None,
        help="Path to an existing session glossary JSON to load and enforce across chunks",
    )
    parser.add_argument(
        "--save-session-glossary",
        nargs="?",
        const="auto",
        default=None,
        help="Save dynamically learned session terms to a JSON file (defaults to <input>_session_terms_<lang>.json if flag provided without path)",
    )
    parser.add_argument(
        "--audit-output",
        default=None,
        help="Path for the verification audit report (defaults to <output>_audit.md when --verify is enabled)",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Compile translated markdown to a publication-ready academic PDF via Typst",
    )
    parser.add_argument(
        "--pdf-output",
        default=None,
        help="Path for the generated academic PDF (defaults to <output>.pdf)",
    )
    parser.add_argument(
        "--paper-size",
        choices=["b5", "a4", "a5", "letter"],
        default="b5",
        help="Paper size for generated PDF (default: b5 / iso-b5, standard Hawza format)",
    )
    parser.add_argument(
        "--book-title",
        default=None,
        help="Textbook title for PDF cover page and headers",
    )
    parser.add_argument(
        "--book-subtitle",
        default=None,
        help="Subtitle for PDF cover page",
    )
    parser.add_argument(
        "--inline-arabic",
        "--bilingual",
        dest="inline_arabic",
        action="store_true",
        help="Render original Arabic paragraphs inline above each translated paragraph (bilingual edition)",
    )
    parser.add_argument(
        "--keep-typ",
        action="store_true",
        help="Retain intermediate .typ source file alongside the PDF",
    )
    parser.add_argument(
        "--list-glossaries",
        action="store_true",
        help="List all built-in languages and domain glossaries",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_glossaries:
        glossaries = list_available_glossaries()
        console.print("[bold cyan]📚 Built-in Glossaries by Language & Domain:[/bold cyan]")
        for lang_code, domains in sorted(glossaries.items()):
            full_name = get_language_name(lang_code)
            console.print(f"  • [bold]{full_name}[/bold] (`{lang_code}`): {', '.join(domains)}")
        sys.exit(0)

    if not args.input:
        parser.error("the following arguments are required: input")

    # --- Slicing configuration ---
    try:
        skip_pages, take_pages = parse_page_range(
            pages_str=args.pages,
            skip=args.skip_pages,
            take=args.take_pages,
        )
    except ValueError as exc:
        console.print(f"[bold red]Error in page range:[/bold red] {exc}")
        sys.exit(1)

    page_tag = format_page_tag(skip_pages, take_pages)

    # --- Handle --list-chapters ---
    if args.list_chapters:
        try:
            full_text = read_file(
                args.input,
                remove_headers_footers=not args.keep_headers,
                separate_footnotes=not args.merge_footnotes,
            )
        except Exception as exc:
            console.print(f"[bold red]Error reading file:[/bold red] {exc}")
            sys.exit(1)

        chapters = detect_chapters(full_text)
        if not chapters:
            console.print(f"[yellow]No chapter or section headings detected in {args.input}.[/yellow]")
        else:
            console.print(f"[bold cyan]📑 Detected Chapters in {args.input}:[/bold cyan]")
            for c in chapters:
                console.print(f"  {c.index}. [bold]{c.title}[/bold] (~{c.word_count:,} words)")
        sys.exit(0)

    target_lang_display = get_language_name(args.lang)

    # --- Header ---
    console.print(
        Panel(
            f"[bold]Arabic Book Translator[/bold]\n"
            f"[dim]Classical Scholastic Arabic → {target_lang_display} ({args.lang}) · Domain: {args.domain.upper()}[/dim]",
            border_style="bright_blue",
        )
    )

    # --- Read input (with page slicing) ---
    slice_desc = f" (pages: {page_tag})" if page_tag else ""
    console.print(f"[cyan]📂 Reading & Normalizing:[/cyan] {args.input}{slice_desc}")
    try:
        source_text = read_file(
            args.input,
            remove_headers_footers=not args.keep_headers,
            separate_footnotes=not args.merge_footnotes,
            skip_pages=skip_pages,
            take_pages=take_pages,
        )
    except (ValueError, FileNotFoundError, ImportError) as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    # --- Chapter filtering (if requested) ---
    chapter_tag = ""
    if args.chapter:
        try:
            source_text, chap_info = extract_chapter(source_text, args.chapter)
            chapter_tag = f"ch{chap_info.index}"
            console.print(
                f"[bold cyan]📑 Sliced to Chapter {chap_info.index}:[/bold cyan] "
                f"{chap_info.title} (~{chap_info.word_count:,} words)"
            )
        except ValueError as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            sys.exit(1)

    char_count = len(source_text)
    console.print(f"[dim]   {char_count:,} characters loaded & normalized[/dim]")

    if not source_text.strip():
        console.print("[bold red]Error:[/bold red] Input file or selected slice is empty.")
        sys.exit(1)

    # --- Session glossary paths ---
    save_session_glossary_path = None
    if args.save_session_glossary == "auto":
        base_name = Path(args.input).with_suffix("").name
        save_session_glossary_path = f"{base_name}_session_terms_{args.lang}.json"
    elif args.save_session_glossary:
        save_session_glossary_path = args.save_session_glossary

    # --- Configure model ---
    cfg = ModelConfig(
        provider=args.provider,
        model=args.model,
        target_lang=args.lang,
        domain=args.domain,
        temperature=args.temperature,
        chunk_size=args.chunk_size,
        overlap_tokens=args.overlap_tokens,
        chunk_delay=args.delay,
        verify=args.verify,
        verifier_model=args.verifier_model,
        verifier_provider=args.verifier_provider,
        session_glossary=args.session_glossary,
        save_session_glossary=save_session_glossary_path,
        inline_arabic=args.inline_arabic,
    )

    # --- Determine Output Paths ---
    slice_parts = [t for t in [chapter_tag, page_tag] if t]
    slice_tag = "_".join(slice_parts) if slice_parts else None
    output_path = args.output or default_output_path(
        args.input,
        lang=args.lang,
        slice_tag=slice_tag,
        bilingual=args.inline_arabic,
    )

    audit_path = None
    if args.verify:
        audit_path = args.audit_output or default_audit_path(output_path)

    # --- Translate ---
    translated = translate(
        text=source_text,
        cfg=cfg,
        glossary_path=args.glossary,
        save_session_glossary=save_session_glossary_path,
        audit_report_path=audit_path,
        dry_run=args.dry_run,
    )

    write_output(translated, output_path)
    console.print(f"[bold green]📝 Output written to:[/bold green] {output_path}")

    # --- Typeset Academic PDF (if requested) ---
    if args.pdf:
        from translator.file_io import default_pdf_path
        from translator.typesetter import compile_markdown_to_pdf, is_typst_available

        if not is_typst_available():
            console.print(
                "[bold yellow]⚠️ Typst is not installed. Install it with 'pip install typst' to enable PDF generation.[/bold yellow]"
            )
        else:
            pdf_path = args.pdf_output or default_pdf_path(output_path)
            book_title = args.book_title or Path(args.input).stem.replace("_", " ").title()
            book_subtitle = args.book_subtitle or f"Hawza Academic Series ({args.domain.upper()})"
            console.print(f"[cyan]📖 Typesetting academic PDF ({args.paper_size.upper()}) via Typst...[/cyan]")
            try:
                compile_markdown_to_pdf(
                    md_text=translated,
                    output_pdf_path=pdf_path,
                    title=book_title,
                    subtitle=book_subtitle,
                    lang=args.lang,
                    domain=args.domain,
                    paper_size=args.paper_size,
                    keep_typ_source=args.keep_typ,
                )
                console.print(f"[bold green]📚 Academic PDF written to:[/bold green] {pdf_path}")
            except Exception as exc:
                console.print(f"[bold red]PDF compilation error:[/bold red] {exc}")



if __name__ == "__main__":
    main()
