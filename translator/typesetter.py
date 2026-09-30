"""Academic Typesetting Module: Convert translated Markdown into publication-ready Hawza PDFs via Typst."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import typst
    _TYPST_AVAILABLE = True
except ImportError:
    typst = None
    _TYPST_AVAILABLE = False


def is_typst_available() -> bool:
    """Return True if the typst compilation engine is installed and available."""
    return _TYPST_AVAILABLE


def normalize_paper_size(size: str) -> str:
    """Normalize paper size name to Typst standard identifier."""
    clean = size.lower().strip()
    if clean in ("b5", "iso-b5", "isob5"):
        return "iso-b5"
    if clean in ("a4", "iso-a4"):
        return "a4"
    if clean in ("a5", "iso-a5"):
        return "a5"
    if clean in ("letter", "us-letter"):
        return "us-letter"
    return "iso-b5"


def markdown_to_typst_content(md_text: str) -> str:
    """Convert academic markdown body text into clean Typst markup."""
    # 1. Parse and extract footnotes: [^1]: Note text or [1]: Note text
    footnotes: Dict[str, str] = {}
    footnote_def_re = re.compile(r"^\[\^?(\d+)\]:\s*(.+)$", re.MULTILINE)
    for match in footnote_def_re.finditer(md_text):
        fn_id, fn_text = match.groups()
        footnotes[fn_id] = fn_text.strip()

    # Remove footnote definitions from body text
    body = footnote_def_re.sub("", md_text)

    # 2. Escape special Typst characters in general text (while protecting markup)
    # Standalone $ and @
    body = re.sub(r"(?<!\\)\$(?!\$)", r"\$", body)
    body = re.sub(r"(?<!\\)@([a-zA-Z])", r"\@\1", body)

    # 3. Convert Footnote references: [^1] or [1] (when matched in definitions)
    for fn_id, fn_content in footnotes.items():
        # Escape any special characters in footnote content
        safe_fn_content = (
            fn_content.replace("$", r"\$")
            .replace("@", r"\@")
            .replace("[", "(")
            .replace("]", ")")
        )
        body = re.sub(rf"\[\^{fn_id}\]", f"#footnote[{safe_fn_content}]", body)
        body = re.sub(rf"\[{fn_id}\]", f"#footnote[{safe_fn_content}]", body)

    # 4. Convert bold and italics
    # Protect bold **text** -> placeholders
    bold_tokens = []

    def bold_repl(m: re.Match) -> str:
        bold_tokens.append(m.group(1))
        return f"__BOLD_TOKEN_{len(bold_tokens)-1}__"

    body = re.sub(r"\*\*(.+?)\*\*", bold_repl, body)

    # Convert single *italic* -> _italic_
    body = re.sub(r"\*([^*\n]+?)\*", r"_\1_", body)

    # Restore bold as Typst *bold*
    for i, token in enumerate(bold_tokens):
        body = body.replace(f"__BOLD_TOKEN_{i}__", f"*{token}*")

    # 5. Line-by-line processing: Headings, blockquotes, lists, dividers
    lines = []
    in_blockquote = False
    blockquote_lines = []
    pending_arabic = None
    pending_tr_lines = []

    def flush_bilingual():
        nonlocal pending_arabic, pending_tr_lines
        if pending_arabic:
            if pending_tr_lines:
                tr_content = "\n".join(pending_tr_lines).strip()
                lines.append(f"\n#bilingual_item[\n  {pending_arabic}\n][\n  {tr_content}\n]\n")
                pending_tr_lines = []
            else:
                lines.append(f"\n#arabic_block[\n  {pending_arabic}\n]\n")
            pending_arabic = None

    def flush_blockquote() -> None:
        nonlocal in_blockquote, blockquote_lines, pending_arabic
        if blockquote_lines:
            content = " ".join(blockquote_lines).strip()
            is_arabic_source = "📜" in content or "الأصل العربي" in content or bool(re.search(r"[\u0600-\u06FF]{6,}", content))
            if is_arabic_source:
                flush_bilingual()
                clean_ar = re.sub(r"^[📜\s*]*\[?الأصل العربي\]?[:\s*]*", "", content).strip()
                pending_arabic = clean_ar
            else:
                flush_bilingual()
                is_sacred = any(
                    kw in content.lower()
                    for kw in ["qur'an", "koran", "hadith", "riwayah", "profeten", "imam", "«", "»", "allah"]
                )
                callout_fn = "quran_callout" if is_sacred else "hawza_callout"
                lines.append(f"\n#{callout_fn}[\n  {content}\n]\n")
            blockquote_lines = []
        in_blockquote = False

    for line in body.splitlines():
        trimmed = line.strip()

        # Blockquote check
        if trimmed.startswith(">"):
            if pending_arabic and pending_tr_lines:
                flush_bilingual()
            in_blockquote = True
            quote_text = trimmed.lstrip("> ").strip()
            blockquote_lines.append(quote_text)
            continue
        elif in_blockquote:
            flush_blockquote()

        # If we have a pending Arabic block, check if this line is part of its translation
        if pending_arabic is not None:
            if not trimmed:
                if pending_tr_lines:
                    flush_bilingual()
                continue
            # If line is a heading or divider, flush bilingual first
            if trimmed.startswith("#") or re.match(r"^[-*_]{3,}$", trimmed):
                flush_bilingual()
            else:
                # Accumulate translation line
                if trimmed.startswith("- "):
                    pending_tr_lines.append(f"- {trimmed[2:].strip()}")
                else:
                    pending_tr_lines.append(trimmed)
                continue

        # Horizontal rules
        if re.match(r"^[-*_]{3,}$", trimmed):
            lines.append("\n#v(0.8em)\n#align(center)[#text(12pt, fill: rgb(\"#1d3557\"))[✦ ✦ ✦]]\n#v(0.8em)\n")
            continue

        # Markdown Headings
        if trimmed.startswith("##### "):
            lines.append(f"===== {trimmed[6:].strip()}")
        elif trimmed.startswith("#### "):
            lines.append(f"==== {trimmed[5:].strip()}")
        elif trimmed.startswith("### "):
            lines.append(f"=== {trimmed[4:].strip()}")
        elif trimmed.startswith("## "):
            lines.append(f"== {trimmed[3:].strip()}")
        elif trimmed.startswith("# "):
            lines.append(f"= {trimmed[2:].strip()}")
        elif re.match(r"^\d+\.\s+", trimmed):
            # Numbered list
            list_item = re.sub(r"^\d+\.\s+", "", trimmed)
            lines.append(f"+ {list_item}")
        elif trimmed.startswith("- "):
            # Bullet list
            lines.append(f"- {trimmed[2:].strip()}")
        else:
            lines.append(line)

    flush_blockquote()
    flush_bilingual()
    return "\n".join(lines)


def build_typst_document(
    body_content: str,
    title: str = "Hawza Curriculum Translation",
    subtitle: str = "Classical Scholastic Arabic Translation",
    lang: str = "da",
    domain: str = "mantiq",
    paper_size: str = "iso-b5",
) -> str:
    """Wrap converted Typst body content in an academic Hawza book template."""
    normalized_paper = normalize_paper_size(paper_size)
    clean_title = title.replace('"', '\\"')
    clean_subtitle = subtitle.replace('"', '\\"')

    domain_labels = {
        "mantiq": "al-Manṭiq al-Islāmī (Logik)",
        "fiqh": "al-Fiqh al-Islāmī (Jurisprudens)",
        "aqida": "al-ʿAqīdah wa-l-Kalām (Teologi)",
        "akhlaaq": "al-Akhlāq wa-l-Irfān (Etik)",
        "tadabbur": "Tadabbur al-Qurʾān (Koranrefleksion)",
        "thaqafa": "al-Thaqāfah al-Islāmiyyah (Kultur og Tænkning)",
        "tarikh": "Tārīkh al-Islām wa-l-Maʿṣūmīn (Historie)",
        "general": "al-Turāth al-Ḥawzawī (Klassisk Arv)",
    }
    domain_display = domain_labels.get(domain.lower(), domain.capitalize())

    return f"""\
#set page(
  paper: "{normalized_paper}",
  margin: (top: 2.5cm, bottom: 2.5cm, inside: 2.5cm, outside: 2.0cm),
  header: context {{
    let page-num = counter(page).get().first()
    if page-num > 1 {{
      if calc.even(page-num) {{
        align(left)[#text(8.5pt, fill: luma(110), font: "Georgia")[*{domain_display}* -- Hawza Studier]]
      }} else {{
        align(right)[#text(8.5pt, fill: luma(110), font: "Georgia")[{clean_title}]]
      }}
      v(-0.4em)
      line(length: 100%, stroke: 0.4pt + luma(200))
    }}
  }},
  footer: context {{
    let page-num = counter(page).get().first()
    if page-num > 1 {{
      align(center)[#text(9pt, fill: luma(100))[-- #counter(page).display("1") --]]
    }}
  }}
)

#set text(
  font: ("Linux Libertine", "Times New Roman", "Georgia", "Amiri", "Traditional Arabic", "Segoe UI"),
  size: 10pt,
  lang: "{lang}"
)
#set par(justify: true, leading: 0.72em, first-line-indent: 1.2em)
#show heading: set text(fill: rgb("#1d3557"), font: ("Georgia", "Times New Roman"))

#let hawza_callout(content) = block(
  width: 100%,
  stroke: (left: 3pt + rgb("#1d3557")),
  fill: rgb("#f6f8fa"),
  inset: (x: 14pt, y: 10pt),
  radius: (right: 4pt),
  spacing: 1.2em,
)[
  #set par(first-line-indent: 0pt)
  #text(style: "italic")[#content]
]

#let quran_callout(content) = block(
  width: 100%,
  stroke: (left: 3pt + rgb("#2d6a4f")),
  fill: rgb("#f2f7f4"),
  inset: (x: 14pt, y: 10pt),
  radius: (right: 4pt),
  spacing: 1.2em,
)[
  #set par(first-line-indent: 0pt)
  #text(fill: rgb("#1b4332"), weight: "medium")[#content]
]

#let arabic_block(content) = block(
  width: 100%,
  fill: rgb("#faf8f5"),
  stroke: (right: 3pt + rgb("#b08968"), rest: 0.5pt + rgb("#e8e2d9")),
  inset: (x: 14pt, y: 10pt),
  radius: (left: 4pt),
  spacing: 0.9em,
)[
  #set text(font: ("Amiri", "Traditional Arabic", "Scheherazade New", "Segoe UI"), size: 11pt, dir: rtl, lang: "ar")
  #set par(justify: true, leading: 0.85em, first-line-indent: 0pt)
  #content
]

#let bilingual_item(arabic, translation) = block(
  width: 100%,
  breakable: false,
  spacing: 1.1em,
)[
  #arabic_block[#arabic]
  #v(0.25em)
  #translation
]

// --- Front Matter / Academic Title Page ---
#align(center)[
  #v(2.0cm)
  #text(10pt, tracking: 2pt, fill: luma(100))[HAWZA ACADEMIC TEXTBOOK SERIES]
  #v(0.8cm)
  #text(22pt, weight: "bold", fill: rgb("#1d3557"))[{clean_title}]
  #v(0.4cm)
  #text(13pt, style: "italic", fill: luma(80))[{clean_subtitle}]
  #v(0.6cm)
  #line(length: 40%, stroke: 1pt + rgb("#1d3557"))
  #v(0.6cm)
  #text(10pt, fill: luma(100))[Scholastisk oversættelse og teologisk audit]
  #v(3.0cm)
]
#pagebreak()

// --- Main Text ---
{body_content}
"""


def compile_markdown_to_pdf(
    md_text: str,
    output_pdf_path: str | Path,
    title: str = "Hawza Curriculum Translation",
    subtitle: str = "Classical Scholastic Arabic Translation",
    lang: str = "da",
    domain: str = "mantiq",
    paper_size: str = "iso-b5",
    keep_typ_source: bool = False,
) -> Path:
    """Compile Markdown translation into a styled academic Hawza PDF using Typst.

    Parameters
    ----------
    md_text:
        The markdown source text.
    output_pdf_path:
        Target path for the compiled PDF file.
    title:
        Title of the textbook/chapter for headers and cover page.
    subtitle:
        Subtitle or volume description.
    lang:
        Language code (e.g. "da", "en").
    domain:
        Hawza academic domain (e.g. "mantiq", "fiqh").
    paper_size:
        Target paper format ("b5", "a4", "a5", "letter").
    keep_typ_source:
        If True, saves the intermediate .typ file alongside the PDF.

    Returns
    -------
    Path
        The path to the successfully generated PDF file.
    """
    if not is_typst_available():
        raise ImportError(
            "The 'typst' package is required for PDF generation. "
            "Install it via: pip install typst"
        )

    out_pdf = Path(output_pdf_path)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    # 1. Convert markdown to Typst body
    typst_body = markdown_to_typst_content(md_text)

    # 2. Wrap in academic Hawza template
    typst_doc = build_typst_document(
        body_content=typst_body,
        title=title,
        subtitle=subtitle,
        lang=lang,
        domain=domain,
        paper_size=paper_size,
    )

    # 3. Write intermediate .typ file
    out_typ = out_pdf.with_suffix(".typ")
    out_typ.write_text(typst_doc, encoding="utf-8")

    # 4. Compile via Typst
    try:
        typst.compile(str(out_typ), output=str(out_pdf))
        logger.info("Successfully compiled academic PDF: %s", out_pdf)
    finally:
        if not keep_typ_source and out_typ.exists():
            try:
                out_typ.unlink()
            except Exception:
                pass

    return out_pdf
