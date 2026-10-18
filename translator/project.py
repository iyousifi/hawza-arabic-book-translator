"""Hawza Translation Project Management.

Enables seamless, incremental translation of large Arabic books (e.g. 100+ pages)
across multiple sessions with persistent terminology memory, antecedent context chaining,
footnote collision prevention, and unified master book compilation (Markdown & Typst PDF).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from translator.config import ModelConfig
from translator.engine import translate
from translator.file_io import read_file
from translator.memory import TerminologyMemory
from translator.typesetter import compile_markdown_to_pdf, is_typst_available

console = Console()


@dataclass
class ProjectSlice:
    """Record of a completed page range or chapter translation slice."""

    start_page: int
    end_page: int
    chunks: int = 0
    char_count: int = 0
    completed_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectSlice":
        return cls(
            start_page=data["start_page"],
            end_page=data["end_page"],
            chunks=data.get("chunks", 0),
            char_count=data.get("char_count", 0),
            completed_at=data.get("completed_at", ""),
        )


@dataclass
class ProjectState:
    """Persistent state and configuration for an academic translation project."""

    version: int = 1
    project_name: str = ""
    source_file: str = ""
    domain: str = "mantiq"
    target_lang: str = "en"
    provider: str = "gemini"
    model: Optional[str] = None
    verify: bool = False
    inline_arabic: bool = True
    strip_harakat: bool = True
    keep_shaddah: bool = False
    arabic_font: str = "amiri"
    paper_size: str = "b5"
    book_title: Optional[str] = None
    book_subtitle: Optional[str] = None
    master_md: str = ""
    master_pdf: str = ""
    session_glossary: str = ""
    audit_file: str = ""
    slices: List[ProjectSlice] = field(default_factory=list)
    next_page: int = 1
    total_pages: Optional[int] = None
    max_footnote_index: int = 0
    trailing_arabic: str = ""
    trailing_target: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["slices"] = [s.to_dict() for s in self.slices]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectState":
        slices = [ProjectSlice.from_dict(s) for s in data.get("slices", [])]
        return cls(
            version=data.get("version", 1),
            project_name=data.get("project_name", ""),
            source_file=data.get("source_file", ""),
            domain=data.get("domain", "mantiq"),
            target_lang=data.get("target_lang", "en"),
            provider=data.get("provider", "gemini"),
            model=data.get("model"),
            verify=data.get("verify", False),
            inline_arabic=data.get("inline_arabic", True),
            strip_harakat=data.get("strip_harakat", True),
            keep_shaddah=data.get("keep_shaddah", False),
            arabic_font=data.get("arabic_font", "amiri"),
            paper_size=data.get("paper_size", "b5"),
            book_title=data.get("book_title"),
            book_subtitle=data.get("book_subtitle"),
            master_md=data.get("master_md", ""),
            master_pdf=data.get("master_pdf", ""),
            session_glossary=data.get("session_glossary", ""),
            audit_file=data.get("audit_file", ""),
            slices=slices,
            next_page=data.get("next_page", 1),
            total_pages=data.get("total_pages"),
            max_footnote_index=data.get("max_footnote_index", 0),
            trailing_arabic=data.get("trailing_arabic", ""),
            trailing_target=data.get("trailing_target", ""),
        )

    def save(self, path: str | Path) -> None:
        """Save project state atomically to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        os.replace(tmp, target)

    @classmethod
    def load(cls, path: str | Path) -> "ProjectState":
        """Load project state from a JSON file."""
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"Project file not found: {target}")
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Footnote & Context Helpers
# ---------------------------------------------------------------------------

def offset_markdown_footnotes(md_text: str, offset: int) -> Tuple[str, int]:
    """Offset footnote references and definitions by offset.

    Parameters
    ----------
    md_text:
        Markdown content containing footnotes [^N] and [^N]:.
    offset:
        Number to add to each footnote index to avoid collisions with prior sections.

    Returns
    -------
    Tuple[str, int]
        (updated_markdown, new_highest_footnote_index)
    """
    if offset <= 0:
        nums = [int(m) for m in re.findall(r"\[\^?(\d+)\]", md_text)]
        return md_text, max(nums) if nums else 0

    max_seen = offset

    def def_sub(m: re.Match) -> str:
        nonlocal max_seen
        num = int(m.group(1))
        new_num = num + offset
        if new_num > max_seen:
            max_seen = new_num
        return f"[^{new_num}]:"

    def ref_sub(m: re.Match) -> str:
        nonlocal max_seen
        num = int(m.group(1))
        new_num = num + offset
        if new_num > max_seen:
            max_seen = new_num
        return f"[^{new_num}]"

    # 1. Update definitions: ^[^1]:
    text = re.sub(r"(?m)^\[\^?(\d+)\]:", def_sub, md_text)
    # 2. Update inline references: [^1] (not followed by colon)
    text = re.sub(r"\[\^(\d+)\](?!:)", ref_sub, text)

    return text, max_seen


def inspect_existing_bilingual_md(md_text: str) -> Dict[str, Any]:
    """Inspect existing bilingual Markdown to extract trailing context and footnote metadata."""
    # Max footnote number
    fn_matches = [int(m) for m in re.findall(r"\[\^?(\d+)\]", md_text)]
    max_fn = max(fn_matches) if fn_matches else 0

    # Strip any trailing standalone footnote section
    cleaned_text = re.sub(
        r"(?i)\n(?:> )?\[(?:الهوامش والتعليقات|Page Footnotes|Noter og kommentarer)[\s\S]*$",
        "",
        md_text,
    ).strip()

    lines = cleaned_text.splitlines()
    last_arabic_lines: List[str] = []
    last_target_lines: List[str] = []

    mode = "target"
    for line in reversed(lines):
        trimmed = line.strip()
        if not trimmed:
            if mode == "arabic" and last_arabic_lines:
                break
            continue
        if trimmed.startswith(">"):
            mode = "arabic"
            last_arabic_lines.insert(0, trimmed.lstrip("> ").strip())
        elif mode == "target":
            last_target_lines.insert(0, trimmed)

    return {
        "max_footnote_index": max_fn,
        "trailing_arabic": " ".join(last_arabic_lines).strip(),
        "trailing_target": " ".join(last_target_lines).strip(),
    }


def get_source_page_count(path: str | Path) -> Optional[int]:
    """Detect total pages if source file is a PDF."""
    p = Path(path)
    if p.suffix.lower() == ".pdf" and p.is_file():
        try:
            import pymupdf

            doc = pymupdf.open(str(p))
            count = len(doc)
            doc.close()
            return count
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Project Operations
# ---------------------------------------------------------------------------

def init_project(
    project_file: str | Path,
    source_file: str | Path,
    domain: str = "mantiq",
    target_lang: str = "en",
    provider: str = "gemini",
    model: Optional[str] = None,
    verify: bool = False,
    inline_arabic: bool = True,
    strip_harakat: bool = True,
    keep_shaddah: bool = False,
    arabic_font: str = "amiri",
    paper_size: str = "b5",
    book_title: Optional[str] = None,
    book_subtitle: Optional[str] = None,
    adopt_md: Optional[str | Path] = None,
    adopt_pages: Optional[str] = None,
    adopt_glossary: Optional[str | Path] = None,
    compile_pdf: bool = True,
) -> ProjectState:
    """Initialize a new translation project manifest."""
    proj_path = Path(project_file)
    if proj_path.suffix.lower() != ".json":
        proj_path = proj_path.with_name(f"{proj_path.stem}.project.json")

    base_dir = proj_path.parent
    base_name = proj_path.stem.replace(".project", "")
    src_path = Path(source_file).resolve()

    if not src_path.is_file():
        raise FileNotFoundError(f"Source file not found: {src_path}")

    title = book_title or base_name.replace("_", " ").title()
    subtitle = book_subtitle or f"Hawza Academic Series ({domain.upper()})"

    master_md = base_dir / f"{base_name}_bilingual_{target_lang}.md"
    master_pdf = base_dir / f"{base_name}_bilingual_{target_lang}.pdf"
    glossary_file = base_dir / f"{base_name}_glossary.json"
    audit_file = base_dir / f"{base_name}_audit.md"

    total_pages = get_source_page_count(src_path)

    state = ProjectState(
        version=1,
        project_name=base_name,
        source_file=str(src_path),
        domain=domain,
        target_lang=target_lang,
        provider=provider,
        model=model,
        verify=verify,
        inline_arabic=inline_arabic,
        strip_harakat=strip_harakat,
        keep_shaddah=keep_shaddah,
        arabic_font=arabic_font,
        paper_size=paper_size,
        book_title=title,
        book_subtitle=subtitle,
        master_md=str(master_md),
        master_pdf=str(master_pdf),
        session_glossary=str(glossary_file),
        audit_file=str(audit_file),
        total_pages=total_pages,
        next_page=1,
    )

    # Adopt an existing translation if requested
    if adopt_md:
        adopt_path = Path(adopt_md)
        if not adopt_path.is_file():
            raise FileNotFoundError(f"File to adopt not found: {adopt_path}")

        md_content = adopt_path.read_text(encoding="utf-8")
        master_md.write_text(md_content, encoding="utf-8")

        inspection = inspect_existing_bilingual_md(md_content)
        state.max_footnote_index = inspection["max_footnote_index"]
        state.trailing_arabic = inspection["trailing_arabic"]
        state.trailing_target = inspection["trailing_target"]

        start_p, end_p = 1, 1
        if adopt_pages:
            m = re.match(r"^(\d+)\s*[-:]\s*(\d+)$", adopt_pages.strip())
            if m:
                start_p, end_p = int(m.group(1)), int(m.group(2))
            else:
                end_p = int(adopt_pages.strip())
                start_p = 1
        state.slices.append(
            ProjectSlice(
                start_page=start_p,
                end_page=end_p,
                char_count=len(md_content),
                completed_at=datetime.now().isoformat(),
            )
        )
        state.next_page = end_p + 1

        if compile_pdf and is_typst_available():
            try:
                pdf_res = compile_markdown_to_pdf(
                    md_text=md_content,
                    output_pdf_path=master_pdf,
                    title=state.book_title or state.project_name,
                    subtitle=state.book_subtitle or f"Hawza Academic Series ({state.domain.upper()})",
                    lang=state.target_lang,
                    domain=state.domain,
                    paper_size=state.paper_size,
                    arabic_font=state.arabic_font,
                    strip_harakat=state.strip_harakat,
                    keep_shaddah=state.keep_shaddah,
                )
                state.master_pdf = str(pdf_res)
            except Exception as exc:
                console.print(f"[bold yellow]Warning: Failed to compile master PDF during adoption: {exc}[/bold yellow]")

    if adopt_glossary:
        adopt_gl = Path(adopt_glossary)
        if adopt_gl.is_file():
            mem = TerminologyMemory.from_file(adopt_gl)
            mem.save_json(glossary_file)

    state.save(proj_path)
    return state


def continue_project(
    project_file: str | Path,
    pages_str: Optional[str] = None,
    take_pages: Optional[int] = None,
    all_pages: bool = False,
    strip_harakat: Optional[bool] = None,
    dry_run: bool = False,
    compile_pdf: bool = True,
) -> ProjectState:
    """Translate the next slice of a project with context chaining and document consolidation."""
    proj_path = Path(project_file)
    state = ProjectState.load(proj_path)

    if strip_harakat is not None:
        state.strip_harakat = strip_harakat

    # 1. Determine target page range
    start_page = state.next_page
    end_page = start_page

    if all_pages:
        start_page = state.next_page
        end_page = state.total_pages or start_page
    elif pages_str:
        m = re.match(r"^(\d+)\s*[-:]\s*(\d+)$", pages_str.strip())
        if m:
            start_page, end_page = int(m.group(1)), int(m.group(2))
        else:
            p_val = int(pages_str.strip())
            start_page, end_page = p_val, p_val
    elif take_pages and take_pages > 0:
        start_page = state.next_page
        end_page = start_page + take_pages - 1
    else:
        # Default to next 10 pages or remainder
        take = 10
        if state.total_pages:
            take = min(take, state.total_pages - start_page + 1)
        end_page = start_page + max(1, take) - 1

    if state.total_pages and end_page > state.total_pages:
        end_page = state.total_pages

    if start_page > end_page:
        raise ValueError(f"Invalid page bounds: start page {start_page} > end page {end_page}")

    if start_page > state.next_page:
        console.print(
            f"[bold yellow]⚠️ Notice: Skipping pages {state.next_page} to {start_page - 1}. "
            f"Translating p{start_page}-{end_page}.[/bold yellow]"
        )

    console.print(
        Panel(
            f"[bold cyan]Project:[/bold cyan] {state.project_name}\n"
            f"[bold cyan]Source:[/bold cyan] {state.source_file}\n"
            f"[bold cyan]Slice:[/bold cyan] Pages {start_page} to {end_page} "
            f"(Total: {end_page - start_page + 1} pages)\n"
            f"[bold cyan]Target:[/bold cyan] {state.target_lang.upper()} ({state.domain.upper()}) | "
            f"Vowels: {'Preserved' if not state.strip_harakat else 'Stripped'}",
            title="📚 Project Continuation",
        )
    )

    # 2. Extract slice text from source file
    skip_pages = start_page - 1
    take_slice = end_page - start_page + 1
    source_slice = read_file(
        state.source_file,
        skip_pages=skip_pages,
        take_pages=take_slice,
    )

    if not source_slice.strip():
        console.print("[bold red]Error: No text extracted from specified page slice.[/bold red]")
        return state

    # 3. Configure translation model
    cfg = ModelConfig(
        provider=state.provider,
        model=state.model,
        target_lang=state.target_lang,
        domain=state.domain,
        verify=state.verify,
        inline_arabic=state.inline_arabic,
        strip_harakat=state.strip_harakat,
        keep_shaddah=state.keep_shaddah,
    )

    # 4. Preload terminology memory from project glossary
    memory = TerminologyMemory()
    gl_path = Path(state.session_glossary)
    if gl_path.is_file():
        memory.load_json(gl_path, source_chunk=0)

    # 5. Translate slice with antecedent context
    slice_output = translate(
        text=source_slice,
        cfg=cfg,
        session_memory=memory,
        save_session_glossary=str(gl_path),
        audit_report_path=state.audit_file if state.verify else None,
        dry_run=dry_run,
        initial_prev_arabic=state.trailing_arabic,
        initial_prev_target=state.trailing_target,
    )

    # 6. Offset footnotes to prevent collision
    renumbered_slice, new_max_fn = offset_markdown_footnotes(
        slice_output, offset=state.max_footnote_index
    )

    # 7. Append to Master Markdown document
    master_path = Path(state.master_md)
    master_path.parent.mkdir(parents=True, exist_ok=True)

    divider = f"\n\n<!-- slice: p{start_page}-{end_page} -->\n\n"
    if master_path.is_file() and master_path.stat().st_size > 0:
        existing_md = master_path.read_text(encoding="utf-8")
        consolidated_md = existing_md.rstrip() + divider + renumbered_slice.strip() + "\n"
    else:
        consolidated_md = renumbered_slice.strip() + "\n"

    master_path.write_text(consolidated_md, encoding="utf-8")
    console.print(f"[bold green]📝 Appended slice to master markdown:[/bold green] {master_path}")

    # 8. Update Trailing Context from the completed slice
    inspection = inspect_existing_bilingual_md(renumbered_slice)
    if inspection["trailing_arabic"]:
        state.trailing_arabic = inspection["trailing_arabic"]
    if inspection["trailing_target"]:
        state.trailing_target = inspection["trailing_target"]

    # 9. Recompile Master PDF
    if compile_pdf and is_typst_available():
        console.print(f"[cyan]📖 Typesetting consolidated master PDF ({state.paper_size.upper()}) via Typst...[/cyan]")
        try:
            pdf_res = compile_markdown_to_pdf(
                md_text=consolidated_md,
                output_pdf_path=state.master_pdf,
                title=state.book_title or state.project_name,
                subtitle=state.book_subtitle or f"Hawza Academic Series ({state.domain.upper()})",
                lang=state.target_lang,
                domain=state.domain,
                paper_size=state.paper_size,
                arabic_font=state.arabic_font,
                strip_harakat=state.strip_harakat,
                keep_shaddah=state.keep_shaddah,
            )
            state.master_pdf = str(pdf_res)
            console.print(f"[bold green]📚 Consolidated PDF written to:[/bold green] {pdf_res}")
        except Exception as exc:
            console.print(f"[bold red]PDF compilation error:[/bold red] {exc}")

    # 10. Update and persist state
    state.slices.append(
        ProjectSlice(
            start_page=start_page,
            end_page=end_page,
            char_count=len(source_slice),
            completed_at=datetime.now().isoformat(),
        )
    )
    state.next_page = end_page + 1
    state.max_footnote_index = new_max_fn
    state.save(proj_path)

    return state


def print_project_status(project_file: str | Path) -> None:
    """Print an academic Rich dashboard summarizing project progress."""
    state = ProjectState.load(project_file)

    table = Table(title=f"Project Dashboard: {state.project_name}", border_style="blue")
    table.add_column("Property", style="bold cyan")
    table.add_column("Value", style="white")

    table.add_row("Source Book", state.source_file)
    table.add_row(
        "Domain & Language",
        f"{state.domain.upper()} → {state.target_lang.upper()}",
    )
    table.add_row(
        "Typography",
        f"Font: {state.arabic_font} | Vowels: {'Preserved' if not state.strip_harakat else 'Stripped'}",
    )

    if state.total_pages:
        completed_pages = sum(s.end_page - s.start_page + 1 for s in state.slices)
        pct = (completed_pages / state.total_pages) * 100
        table.add_row(
            "Progress",
            f"{completed_pages} / {state.total_pages} pages ({pct:.1f}%)",
        )
    table.add_row("Next Page", str(state.next_page))

    # Completed slices
    slices_str = ", ".join(f"p{s.start_page}-{s.end_page}" for s in state.slices) or "None"
    table.add_row("Completed Slices", slices_str)

    # Master files
    table.add_row("Master Markdown", state.master_md)
    table.add_row("Master PDF", state.master_pdf)

    # Glossary count
    gl_path = Path(state.session_glossary)
    terms_count = 0
    if gl_path.is_file():
        try:
            mem = TerminologyMemory.from_file(gl_path)
            terms_count = len(mem)
        except Exception:
            pass
    table.add_row("Learned Terminology", f"{terms_count} scholastic terms")

    console.print(table)
    console.print(
        f"\n[dim]To continue translating, run:[/dim]\n"
        f"[bold green]py -3.11 -m translator project run \"{project_file}\" --take-pages 15[/bold green]\n"
    )
