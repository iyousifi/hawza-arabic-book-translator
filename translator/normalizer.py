"""Smart Arabic Text Normalizer and Layout-Aware Extraction for Classical Scholastic Texts."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None


# ---------------------------------------------------------------------------
# Unicode Character Sets & Patterns
# ---------------------------------------------------------------------------

# Arabic diacritics (Tashkeel / Harakat)
_HARAKAT_RE = re.compile(
    r"[\u0617-\u061A\u064B-\u0652\u0656-\u065F\u0670]"
)

# Invisible directional controls and zero-width artifacts
_BIDI_CONTROLS_RE = re.compile(
    r"[\u200E\u200F\u202A-\u202E\u2066-\u2069\uFEFF\u200B\u200C\u200D]"
)

# Arabic Tatweel / Kashida (decorative elongation)
_TATWEEL_RE = re.compile(r"\u0640+")

# Footnote line rules (e.g. ________ or --------)
_FOOTNOTE_RULE_RE = re.compile(r"^[\s_—–-]{3,}\s*$")

# Footnote markers, e.g. (1), (١), [1], [١], (*), 1., 1-, ١., ١-, 1», .1 at start or end of block
_FOOTNOTE_MARKER_RE = re.compile(
    r"([(\[]\s*[\d٠-٩*†]+\s*[)\]]"
    r"|\*"
    r"|(?<![\u0600-\u06FF])[\d٠-٩]+[\-.)]"
    r"|(?<![\u0600-\u06FF])[\d٠-٩]+\s*»"
    r"|[.،]\s*[\d٠-٩]+\b)"
)

# Pure page number pattern (Western or Eastern Arabic digits)
_PAGE_NUM_RE = re.compile(r"^\s*[-–—]?\s*[\d٠-٩]+\s*[-–—]?\s*$")

# Legacy 1990s Quranic/Calligraphic font artifacts (e.g. wmNaskhQurani93, MacNaskh)
# These fonts mapped decorative connector swashes and ligatures into Latin Extended codepoints.
_UNCONDITIONAL_QURANI_ARTIFACTS = "ĀăąćĊċđĒēĔĕĖĚĩĪĭĮıĳŅŊŋŌōřŚƑƔƕƗƜƝƸƺƻǌǍǕǢȈȝȤ͜͝͞▪\ufffd"
_QURANI_FONT_ARTIFACTS_RE = re.compile(f"[{re.escape(_UNCONDITIONAL_QURANI_ARTIFACTS)}]")

_CONDITIONAL_LATIN_ARTIFACTS = "ø÷äöüþúùûóõÿ"
_CONDITIONAL_LATIN_ARTIFACTS_RE = re.compile(
    rf"[{re.escape(_CONDITIONAL_LATIN_ARTIFACTS)}](?=[\u0600-\u06FF])|(?<=[\u0600-\u06FF])[{re.escape(_CONDITIONAL_LATIN_ARTIFACTS)}]"
)


# ---------------------------------------------------------------------------
# Reversal (Bidi / Visual RTL) Detection & Repair
# ---------------------------------------------------------------------------

def is_arabic_text_reversed(text: str) -> bool:
    """Detect if Arabic text has been extracted in reversed (visual LTR) order."""
    words = re.findall(r"[\u0600-\u06FF]+", text)
    if not words:
        return False

    # 1. Taa Marbutah (ة) at the start of a word is grammatically impossible in Arabic
    if any(w.startswith("ة") for w in words):
        return True

    # 2. Definite article 'ال' check vs reversed 'لا' at word ends
    starts_al = sum(1 for w in words if w.startswith("ال"))
    ends_la = sum(1 for w in words if w.endswith("لا"))

    if ends_la > starts_al and ends_la > 0:
        return True

    return False


def fix_reversed_arabic_line(line: str) -> str:
    """Reverse a line back to logical RTL order and fix bracket/digit orientation."""
    rev = line[::-1]
    # Invert paired punctuation back to correct logical orientation
    bracket_swap = str.maketrans("()[]{}«»<>", ")(][}{»«><")
    rev = rev.translate(bracket_swap)
    # Digits are already written LTR, so line[::-1] flipped them - flip them back
    rev = re.sub(r"[\d٠-٩]+", lambda m: m.group(0)[::-1], rev)
    return rev


def detect_and_fix_reversed_arabic(text: str) -> str:
    """Detect if an Arabic text is reversed and restore it to logical reading order."""
    if not text:
        return ""

    if is_arabic_text_reversed(text):
        lines = text.split("\n")
        fixed_lines = [fix_reversed_arabic_line(l) for l in lines]
        return "\n".join(fixed_lines)

    return text


def fix_font_ligature_reversals(text: str) -> str:
    """Repair font-specific glyph and ligature reversal artifacts commonly found in Arabic PDFs.

    Classical Arabic typesetting fonts (such as Lotus, Swissra, or Traditional Arabic) often have
    ligature glyphs whose ToUnicode CMap entries were stored in visual or inverted byte order.
    This function restores the proper phonetic/logical character order for:
      - Standalone and prefixed prepositions (في, على, إلى, كما, بين)
      - Alef-Lam ligatures (الأ, الإ, الآ, الا)
      - Descending/hanging Lam ligatures with Jim, Hah, Khah, Meem, Heh (الج, الح, الخ, الم, اله)
      - Nisba/adjective endings (-الي, -اني, -افي) and reversed -يرة endings (كبيرة, كثيرة, etc.)
      - Systemic typographic reversals in Hawza logic terminology (اجتماع, استدلال, أفلاطون, الشيء, تماما, إلخ).
    """
    if not text:
        return ""

    # 1. Standalone prepositions & particles:
    text = re.sub(r"(?<![\u0600-\u06FF])([وف])?يف(?![\u0600-\u06FF])", r"\1في", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلكب])?عىل(?![\u0600-\u06FF])", r"\1على", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلكب])?إىل(?![\u0600-\u06FF])", r"\1إلى", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وف])?كام(?![\u0600-\u06FF])", r"\1كما", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وف]|ما)?بني(?![\u0600-\u06FF])", r"\1بين", text)

    # 2. Definite article (ال) ligature reversals:
    text = text.replace("األ", "الأ").replace("اإل", "الإ").replace("اآل", "الآ")
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلكب]?|[لب]ل)اال", r"\1الا", text)

    # 3. Classical hanging ligatures with Lam (ج, ح, خ, م, ه):
    text = re.sub(r"(?<![\u0600-\u06FF])([وفبك]?|[لب]ل)امل", r"\1الم", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفبك]?|[لب]ل)اجل", r"\1الج", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفبك]?|[لب]ل)احل", r"\1الح", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفبك]?|[لب]ل)اخل", r"\1الخ", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفبك]?|[لب]ل)اهل(?![\u0600-\u06FF])", r"\1اله", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفبك]?|[لب]ل)اهل([ء-ي])", lambda m: m.group(0) if m.group(2) == "ك" else m.group(1) + "اله" + m.group(2), text)

    # 4. Final nisba & adjective letter reversals:
    text = re.sub(r"([ء-ي]+)ايل\b", r"\1الي", text)
    text = re.sub(r"([ء-ي]+)اين\b", lambda m: m.group(1) + "اني" if m.group(1) in ("الث", "والث", "اليون", "الإنس", "الوجـد", "المع", "ث") else m.group(0), text)
    text = re.sub(r"([ء-ي]+)ايف\b", r"\1افي", text)

    # 5. Adjectives ending in رية that should be يرة:
    roots = ["كب", "كث", "صغ", "قص", "خط", "مس", "أخ", "اخ", "عس", "وف", "شه", "جز", "بص", "يس"]
    for r in roots:
        text = re.sub(rf"(?<![\u0600-\u06FF])((?:ال|وال|فال|بال|كال|و|ف|ب|ل|ك)?){r}رية(?![\u0600-\u06FF])", rf"\1{r}يرة", text)

    # 6. Specific systemic typographic reversals in classical text:
    text = text.replace("جتامع", "جتماع")
    text = text.replace("ثالثة", "ثلاثة").replace("الثالثة", "الثلاثة")
    text = text.replace("تارخي", "تاريخ")
    text = text.replace("أفالطون", "أفلاطون")
    text = text.replace("متزج", "تمزج")
    text = text.replace("أخريا", "أخيرا").replace("اخريا", "أخيرا")
    text = text.replace("اإلجيايب", "الإيجابي").replace("الإجيايب", "الإيجابي").replace("إجيايب", "إيجابي")
    text = text.replace("مالحظ", "ملاحظ")
    text = text.replace("االجحاف", "الإجحاف")
    text = re.sub(r"([ء-ي]+)وهلا\b", r"\1ولها", text)
    text = re.sub(r"([ء-ي]+)تهام\b", r"\1تهما", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلكسي]?|يت)خيت", r"\1يخت", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلكسي]?|يت)خيلف", r"\1يخلف", text)
    text = re.sub(r"(?<![\u0600-\u06FF])((?:[وفلبك]?(?:ال)?))رضوري", r"\1ضروري", text)
    text = re.sub(r"(?<![\u0600-\u06FF])((?:[وفلبك]?(?:ال)?))يشء", r"\1شيء", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلكب])?متاما\b", r"\1تماما", text)
    text = re.sub(r"(?<![\u0600-\u06FF])((?:[وفلبك]?(?:ال)?))رصح(?![\u0600-\u06FF])", r"\1صرح", text)
    text = re.sub(r"\bترشح\b", "تشرح", text)
    text = re.sub(r"\bفرشح\b", "فشرح", text)
    text = re.sub(r"\bحتدد\b", "تحدد", text)
    text = re.sub(r"\bسري\s+(الفكر|التاريخ|الأحداث|المنطق)", r"سير \1", text)
    text = re.sub(
        r"\b([وفب]?)ثالث\s+(زوايا|خطوات|درجات|مراحل|قواعد|أسئلة|عوامل|حقائق|ملاحظات|مالحظات|أقسام|مرات|وسائل|مفكرين|حالات|علامات|أيام|سنين|سنوات)",
        r"\1ثلاث \2",
        text,
    )

    # 7. Legacy Calligraphic / Quranic font artifacts (wmNaskhQurani93) & intra-word spacing:
    text = _QURANI_FONT_ARTIFACTS_RE.sub("", text)
    text = _CONDITIONAL_LATIN_ARTIFACTS_RE.sub("", text)
    text = re.sub(r"(?<=[\u0600-\u06FF\s])(?:nj|lj)(?=[\u0600-\u06FF\s])", "", text)
    # Reconnect Arabic diacritics separated by space from their base letters
    text = re.sub(r"(?<=[\u0600-\u06FF])\s+([\u064B-\u065F\u0670])", r"\1", text)

    # Reconstitute fragmented Quranic verses commonly typeset with wmNaskhQurani93:
    text = re.sub(r"خ\s*َلَ\s*ق\s*َ", "خَلَقَ", text)
    text = re.sub(r"ال\s*إ\s*ِ\s*نْسَا\s*ن\s*َ", "الإِنْسَانَ", text)
    text = re.sub(r"ال\s*إ\s*ِ\s*نْسا\s*ن\s*ُ", "الإِنْسَانُ", text)
    text = re.sub(r"مِ\s*ن\s*ْ", "مِنْ", text)
    text = re.sub(r"عَلَ\s*ق\s*ٍ", "عَلَقٍ", text)
    text = re.sub(r"ا\s*قْرَ\s*أْ?", "اقْرَأْ", text)
    text = re.sub(r"رَ\s*ب(?:[\u0651\u064F]{2}|[\u0651\u064F])?كَ", "رَبُّكَ", text)
    text = re.sub(r"رَ\s*ب(?:[\u0651\u0650]{2}|[\u0651\u0650])?كَ", "رَبِّكَ", text)
    text = re.sub(r"(?<![\u0600-\u06FF])[ْـ]\s*(?=بِاسْمِ)", "", text)
    text = re.sub(r"ال\s*أَكْرَمُ", "الأَكْرَمُ", text)
    text = re.sub(r"الَّ\s*ذ\s*ِ\s*ي", "الَّذِي", text)
    text = re.sub(r"الَّ\s*ذ\s*ِي\s*ن\s*َ", "الَّذِينَ", text)
    text = re.sub(r"بِالْ\s*ق\s*َلَمِ", "بِالْقَلَمِ", text)
    text = re.sub(r"ي\s*َعْلَمْ", "يَعْلَمْ", text)
    text = re.sub(r"ي\s*َ\s*خْ\s*ش\s*َى", "يَخْشَى", text)
    text = re.sub(r"عِ\s*ب\s*ادِهِ", "عِبَادِهِ", text)
    text = re.sub(r"إ\s*ِ\s*نَّما", "إِنَّمَا", text)
    text = re.sub(r"يَا\s+أَ\s*يُّهَا", "يَا أَيُّهَا", text)
    text = re.sub(r"آمَ\s*ن\s*ُوا", "آمَنُوا", text)
    text = re.sub(r"ا\s*تَّ\s*ق\s*ُوا", "اتَّقُوا", text)
    text = re.sub(r"كُو\s*نُوا", "كُونُوا", text)
    text = re.sub(r"الصَّادِ\s*قِ\s*ي\s*ن\s*َ", "الصَّادِقِينَ", text)
    text = re.sub(r"لِ\s*ي\s*َ\s*ت\s*َ\s*ف\s*َ\s*ق\s*َّهُوا", "لِيَتَفَقَّهُوا", text)
    text = re.sub(r"فِ\s*ي\s+الدِّ\s*ي\s*ن\s*ِ", "فِي الدِّينِ", text)
    text = re.sub(r"وَلِ\s*يُ\s*ن\s*ْ\s*ذ\s*ِرُوا", "وَلِيُنْذِرُوا", text)
    text = re.sub(r"قَوْمَهُمْ", "قَوْمَهُمْ", text)
    text = re.sub(r"إ\s*ِ\s*ذ\s*َا", "إِذَا", text)
    text = re.sub(r"رَ\s*ج\s*َعُوا", "رَجَعُوا", text)
    text = re.sub(r"إ\s*ِلَ\s*يْهِمْ", "إِلَيْهِمْ", text)
    text = re.sub(r"ي\s*َحْ\s*ذ\s*َرُو\s*ن\s*َ?", "يَحْذَرُونَ", text)
    text = re.sub(r"فَلْ\s*يَ\s*ن\s*ْ\s*ظ\s*ُرِ", "فَلْيَنْظُرِ", text)
    text = re.sub(r"طَعامِهِ", "طَعَامِهِ", text)
    text = re.sub(r"يَسْ\s*ت\s*َمِعُو\s*ن\s*َ", "يَسْتَمِعُونَ", text)
    text = re.sub(r"الْ\s*ق\s*َوْلَ", "الْقَوْلَ", text)
    text = re.sub(r"فَ\s*يَ\s*تَّ\s*بِعُو\s*ن\s*َ", "فَيَتَّبِعُونَ", text)
    text = re.sub(r"أَحْسَ\s*نَهُ", "أَحْسَنَهُ", text)

    # 8. Swissra font ligature fixes and broken decomposed Lam-Alef words:
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلكب]?)ال(?:ك| )فايئ\b", r"\1الكفائي", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلكب]?)ال(?:ع| )يي\b", r"\1العيني", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلكب]?)(?:اإلس|الإس)\s+اام\b", r"\1الإسلام", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلكب]?)(?:املعام|المعام)\s+اات\b", r"\1المعاملات", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلكب]?)(?:االبت|الابت)\s+ااء\b", r"\1الابتلاء", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلكب]?)الكر\s+مي\b", r"\1الكريم", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلكب]?)خب\s+ااف\b", r"\1خلاف", text)
    text = re.sub(r"\bف\s+اا\b", "فلا", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلكب]?)ال\s+اازِمُ?\b", r"\1اللازم", text)
    text = re.sub(r"\bرَجُ\s+اانِ\b", "رجلان", text)
    text = re.sub(r"\bجَاهِ\s+ااً\b", "جاهلاً", text)
    text = re.sub(r"\bجَهْ\s+ااً\b", "جهلاً", text)
    text = re.sub(r"\bوَجِ\s+ااً\b", "وجلاً", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلكب]?)الإخ\s+ااص\b", r"\1الإخلاص", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلكب]?)المكال\b", r"\1الكمال", text)
    text = re.sub(r"\bمب(ستوى|ختلف|قدار|ا|عنى|وجب|قتضى)\b", r"بم\1", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفل]?)ميكن(ه?)\b", r"\1يمكن\2", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفل]?)ميوت\b", r"\1يموت", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلبك]?)حتقيق\b", r"\1تحقيق", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلبك]?)حتصيل\b", r"\1تحصيل", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلبك]?)إال(?![\u0600-\u06FF])", r"\1إلا", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وف]?)هبا(?![\u0600-\u06FF])", r"\1بها", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وف]?)هلا(?![\u0600-\u06FF])", r"\1لها", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وف]?)هنم(?![\u0600-\u06FF])", r"\1منهم", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وف]?)عهنم(?![\u0600-\u06FF])", r"\1عنهم", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وف]?)مهنا(?![\u0600-\u06FF])", r"\1منها", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلبك]?)أدلهتا\b", r"\1أدلتها", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلبك]?)أمهية\b", r"\1أهمية", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلبك]?)صحّهتا\b", r"\1صحتها", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلبك]?)الشهات\b", r"\1الشبهات", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلبك]?)الإشكاالت\b", r"\1الإشكالات", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلبك]?)اصطالحات\b", r"\1اصطلاحات", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلبك]?)حديهثم\b", r"\1حديثهم", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلبك]?)(?:الُْؤْمِنِنَ|المؤمنن)\b", r"\1المؤمنين", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلبك]?)الُْسَنِْ\b", r"\1الحسن", text)
    text = re.sub(r"(?<![\u0600-\u06FF])([وفلبك]?)عَيِِّ\b", r"\1علي", text)
    text = re.sub(r"\bمعىً\b", "معنىً", text)
    text = re.sub(r"\bالقلي هلل\b", "القلبي لله", text)
    text = re.sub(r"\bمل\s+(يكن|يتمكّن|يدرك|يسبق|يقصد|يراع|يمكن|تكن)", r"لم \1", text)

    return text


# ---------------------------------------------------------------------------
# Text Normalization
# ---------------------------------------------------------------------------

def normalize_arabic_text(
    text: str,
    strip_harakat: bool = False,
    standardize_digits: bool = False,
    fix_reversals: bool = True,
) -> str:
    """Normalize Arabic text extracted from digital files.

    Operations:
    1. Resolves Arabic Presentation Forms (A & B) to standard canonical Arabic via NFKD.
    2. Detects and corrects reversed (visual LTR) Arabic order.
    3. Strips calligraphic Tatweel/Kashida (ـ).
    4. Cleans invisible Bidi marks and zero-width spaces.
    5. Recomposes canonical diacritics via NFC.
    6. Normalizes Arabic punctuation spacing and line breaks.
    7. Optionally strips decorative Tashkeel or standardizes numerals.
    """
    if not text:
        return ""

    # Step 0: Strip legacy calligraphic/Quranic font artifacts before decomposition
    text = _QURANI_FONT_ARTIFACTS_RE.sub("", text)
    text = _CONDITIONAL_LATIN_ARTIFACTS_RE.sub("", text)

    # Step 1: Decompose presentation forms (e.g. ﻼ \uFEFB, ﷲ \uFDF2) into standard Arabic
    decomposed = unicodedata.normalize("NFKD", text)

    # Step 2: Detect and correct reversed Arabic text if present
    if fix_reversals:
        decomposed = detect_and_fix_reversed_arabic(decomposed)

    # Step 3: Strip tatweel / kashida
    cleaned = _TATWEEL_RE.sub("", decomposed)

    # Step 4: Strip invisible Bidi directional markers and control characters
    cleaned = _BIDI_CONTROLS_RE.sub("", cleaned)
    cleaned = cleaned.replace("\u00a0", " ")  # Non-breaking space

    # Step 5: Re-compose to NFC canonical Unicode
    recomposed = unicodedata.normalize("NFC", cleaned)

    # Recompose decomposed Alef/Waw/Yeh + combining hamza/madda to standard precomposed letters
    recomposed = (
        recomposed
        .replace("\u0627\u0655", "\u0625")
        .replace("\u0655\u0627", "\u0625")
        .replace("\u0627\u0654", "\u0623")
        .replace("\u0654\u0627", "\u0623")
        .replace("\u0627\u0653", "\u0622")
        .replace("\u0653\u0627", "\u0622")
        .replace("\u0648\u0654", "\u0624")
        .replace("\u064A\u0654", "\u0626")
        .replace("\u0649\u0654", "\u0626")
    )

    # Step 6: Fix font-specific glyph and ligature reversal artifacts
    if fix_reversals:
        recomposed = fix_font_ligature_reversals(recomposed)

    # Step 7: Optional removal of harakat (if requested for search/matching)
    if strip_harakat:
        recomposed = _HARAKAT_RE.sub("", recomposed)

    # Step 7: Optional numeral standardization (Eastern Arabic ٠-٩ to Western 0-9)
    if standardize_digits:
        eastern_to_western = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
        recomposed = recomposed.translate(eastern_to_western)

    # Step 8: Clean whitespace and line endings
    recomposed = recomposed.replace("\r\n", "\n").replace("\r", "\n")
    recomposed = re.sub(r"[ \t]+", " ", recomposed)
    recomposed = re.sub(r"\n{3,}", "\n\n", recomposed)

    # Step 9: Clean line-start punctuation artifacts (e.g. leading commas '،', inverted colons ':الدرس')
    recomposed = re.sub(r"(?m)^:\s*([\u0600-\u06FF\s]+)$", r"\1:", recomposed)
    recomposed = re.sub(r"(?m)^[،,:]\s*", "", recomposed)

    return recomposed.strip()


# ---------------------------------------------------------------------------
# Layout-Aware Block Extraction
# ---------------------------------------------------------------------------

@dataclass
class PageBlock:
    """Represents a text block with spatial coordinates on a page."""
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float = 12.0
    is_footnote: bool = False
    is_header_or_footer: bool = False
    is_margin_rubric: bool = False


def extract_footnote_divider_lines(page: Any) -> List[Tuple[float, float, float]]:
    """Extract vector horizontal divider lines in the bottom 45% of the page.

    Returns a list of tuples: (x0, x1, y)
    """
    lines: List[Tuple[float, float, float]] = []
    if page is None or not hasattr(page, "get_drawings"):
        return lines
    try:
        height = page.rect.height
        for d in page.get_drawings():
            for item in d.get("items", []):
                if item[0] == "l":  # line
                    p1, p2 = item[1], item[2]
                    # Strictly horizontal line within 2.0pt tolerance
                    if abs(p1.y - p2.y) < 2.0:
                        y = (p1.y + p2.y) / 2.0
                        # Must be in the lower 45% of the page
                        if y > height * 0.55:
                            x0 = min(p1.x, p2.x)
                            x1 = max(p1.x, p2.x)
                            # Must have significant width (> 35pt)
                            if (x1 - x0) > 35.0:
                                lines.append((x0, x1, y))
    except Exception:
        pass
    return lines


def classify_page_blocks(
    blocks: List[Any],
    page_width: float,
    page_height: float,
    remove_headers_footers: bool = True,
    separate_footnotes: bool = True,
    exclude_margin_rubrics: bool = True,
    return_rubrics: bool = False,
    footnote_divider_lines: Optional[List[Tuple[float, float, float]]] = None,
) -> Tuple[List[str], List[str]] | Tuple[List[str], List[str], List[str]]:
    """Classify extracted PDF blocks into body text and footnotes.

    Parameters
    ----------
    blocks:
        List of PyMuPDF text blocks (x0, y0, x1, y1, text, block_no, block_type, [font_size]).
    page_width:
        Width of the page.
    page_height:
        Height of the page.
    remove_headers_footers:
        Whether to filter out running headers (<7.5% page height) and solitary footers (>92.5%).
    separate_footnotes:
        Whether to identify and separate bottom commentary/footnotes.
    exclude_margin_rubrics:
        Whether to exclude vertical margin tabs and side ribbons (e.g. 'مدخل') from body text.
    return_rubrics:
        Whether to return the list of cleaned margin rubric titles as a third tuple element.
    footnote_divider_lines:
        Optional list of (x0, x1, y) horizontal line coordinates indicating column-aware footnote boundaries.

    Returns
    -------
    Tuple[List[str], List[str]] | Tuple[List[str], List[str], List[str]]
        (body_blocks, footnote_blocks) or (body_blocks, footnote_blocks, rubric_texts)
    """
    classified: List[PageBlock] = []

    # Calculate typical body font size if font sizes are available in blocks
    body_sizes = [
        b[7] for b in blocks
        if len(b) > 7 and (page_height * 0.15 < b[1] < page_height * 0.65) and b[7] > 0
    ]
    typical_body_size = (sum(body_sizes) / len(body_sizes)) if body_sizes else 0.0

    for b in blocks:
        # b[6] is block_type: 0 = text, 1 = image
        if len(b) <= 6 or b[6] != 0:
            continue

        raw_text = b[4].strip()
        if not raw_text:
            continue

        norm_text = normalize_arabic_text(raw_text)
        if not norm_text:
            continue

        x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
        font_size = b[7] if len(b) > 7 else 12.0

        block = PageBlock(
            text=norm_text,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            font_size=font_size,
        )

        # 1. Header detection: top 7.5% of page
        if remove_headers_footers and y1 < page_height * 0.075:
            block.is_header_or_footer = True

        # 2. Footer detection: bottom 7.5% of page and pure page number
        if remove_headers_footers and y0 > page_height * 0.925 and _PAGE_NUM_RE.match(norm_text):
            block.is_header_or_footer = True

        # 3. Footnote line rule
        if _FOOTNOTE_RULE_RE.match(norm_text):
            continue  # omit separator rule line itself

        # 4. Vertical margin tab / rubric detection:
        bw = x1 - x0
        bh = y1 - y0
        in_outer_margin = (x0 > page_width * 0.86) or (x1 < page_width * 0.14)
        is_tall_vertical_ribbon = (bh > 1.8 * bw) and (bh > 35)
        is_margin_rubric_keyword = any(
            kw in norm_text for kw in ("مدخل", "تمهيد", "تنبيه", "تذكرة", "إشارة", "إضاءة")
        )

        if exclude_margin_rubrics and in_outer_margin and (is_tall_vertical_ribbon or is_margin_rubric_keyword):
            block.is_margin_rubric = True

        # 5. Column-aware footnote detection via vector divider lines:
        if separate_footnotes and footnote_divider_lines:
            for lx0, lx1, ly in footnote_divider_lines:
                # Block is below the horizontal line in the same column/zone
                if y0 >= ly - 4.0 and not (x1 < lx0 - 15.0 or x0 > lx1 + 15.0):
                    block.is_footnote = True
                    break

        classified.append(block)

    # Sort blocks: For RTL texts, right-hand columns come before left-hand columns
    is_multi_column = _detect_multi_column(classified, page_width)

    if is_multi_column:
        # Multi-column sorting: right column (higher x0) first, then by y0
        classified.sort(key=lambda blk: (0 if blk.x0 > page_width * 0.45 else 1, blk.y0))
    else:
        # Single column: sort by vertical bands (12pt height tolerance) and RTL descending x0
        classified.sort(key=lambda blk: (round(blk.y0 / 12.0), -blk.x0))

    # Footnote boundary detection:
    footnote_started = False
    body_blocks: List[str] = []
    footnote_blocks: List[str] = []
    rubric_texts: List[str] = []

    for blk in classified:
        if blk.is_margin_rubric:
            clean_rubric = "".join(c for c in blk.text if c not in ("ـ", " ", "\n", "\r", "\t"))
            if clean_rubric and clean_rubric not in rubric_texts:
                rubric_texts.append(clean_rubric)
            continue

        if blk.is_header_or_footer:
            continue

        # If not already flagged by vector divider lines, test textual markers or font size (only when no vector lines exist)
        if separate_footnotes and not blk.is_footnote and not footnote_divider_lines:
            in_bottom_zone = blk.y0 > page_height * 0.65
            has_marker = bool(
                _FOOTNOTE_MARKER_RE.search(blk.text[:25])
                or _FOOTNOTE_MARKER_RE.search(blk.text[-25:])
            )
            is_small_citation = bool(
                typical_body_size > 13.0
                and blk.font_size > 0
                and blk.font_size < typical_body_size * 0.8
                and any(kw in blk.text for kw in ("ص", "ج", "ح", "طبعة", "المصدر", "الكافي", "بحار", "وسائل"))
            )

            # In single-column pages, once a footnote starts in the bottom zone, succeeding bottom blocks follow.
            # In multi-column pages, do NOT cascade footnote_started across columns.
            if in_bottom_zone and (has_marker or is_small_citation or (not is_multi_column and footnote_started)):
                footnote_started = True
                blk.is_footnote = True

        if blk.is_footnote:
            footnote_blocks.append(blk.text)
        else:
            body_blocks.append(blk.text)

    # Reassemble blocks that were fragmented across line wraps in the PDF
    merged_body = merge_fragmented_blocks(body_blocks)
    if return_rubrics:
        return merged_body, footnote_blocks, rubric_texts
    return merged_body, footnote_blocks


def merge_fragmented_blocks(blocks: List[str]) -> List[str]:
    """Merge PDF text blocks that were split mid-sentence or mid-paragraph.

    If a block does not end with terminal punctuation (. ! ؟ :) and the next
    block is not a section heading or bullet point, merge them into a single paragraph.
    """
    merged: List[str] = []
    heading_re = re.compile(
        r"^(?:#+|\d+[\-.)]|[-•*]\s+|[:\s]*(?:المقصد|الباب|الفصل|المبحث|المسألة|الدرس|العنوان|كلمة))"
    )
    terminal_re = re.compile(r"[.!:؟]\s*$")
    list_end_re = re.compile(r"[-–—.]?\s*[\d٠-٩]+[-–—.]?\s*$")
    list_start_re = re.compile(r"^[\s.ـ]*[-–—]?\s*[\d٠-٩]+[\-.)]")

    for b in blocks:
        clean = b.strip()
        if not clean:
            continue
        if not merged:
            merged.append(clean)
            continue
        prev = merged[-1]

        # 1. Header label "العنوان" followed by title text
        if prev in ("العنوان", "عنوان", "الموضوع") and len(clean) < 60:
            merged[-1] = f"{prev}: {clean}"
            continue

        prev_has_terminal = (
            bool(terminal_re.search(prev))
            or bool(list_end_re.search(prev))
        )
        prev_is_heading = (
            bool(heading_re.match(prev))
            or (len(prev) < 40 and not prev.endswith((".", "،")))
        )
        curr_is_heading = bool(heading_re.match(clean))
        curr_is_list = bool(list_start_re.match(clean))

        if not prev_has_terminal and not prev_is_heading and not curr_is_heading and not curr_is_list:
            merged[-1] = f"{prev} {clean}"
        else:
            merged.append(clean)
    return merged


def _detect_multi_column(blocks: List[PageBlock], page_width: float) -> bool:
    """Detect if page blocks form a 2-column layout."""
    if len(blocks) < 4:
        return False

    midpoint = page_width * 0.5
    has_right = any(b.x0 > midpoint and b.x1 > page_width * 0.7 for b in blocks)
    has_left = any(b.x1 < midpoint and b.x0 < page_width * 0.3 for b in blocks)
    has_spanning = any(b.x0 < page_width * 0.25 and b.x1 > page_width * 0.75 for b in blocks)

    return has_right and has_left and not has_spanning


# ---------------------------------------------------------------------------
# PDF Reader using Smart Normalization
# ---------------------------------------------------------------------------

def _extract_blocks_with_styling(page: Any) -> List[Tuple[float, float, float, float, str, int, int]]:
    """Extract page blocks while preserving royal blue (0x24408f) span styling."""
    try:
        dict_data = page.get_text("dict")
        blocks: List[Tuple[float, float, float, float, str, int, int]] = []
        for idx, b in enumerate(dict_data.get("blocks", [])):
            if "lines" not in b:
                continue
            x0, y0, x1, y1 = b["bbox"]
            line_strings = []
            font_sizes = []
            for l in b["lines"]:
                spans = l.get("spans", [])
                line_parts = []
                for s in spans:
                    t = s.get("text", "")
                    if not t:
                        continue
                    if t.strip():
                        font_sizes.append(s.get("size", 12.0))
                    t = fix_font_ligature_reversals(t)
                    color = s.get("color", 0)
                    if color == 0x24408f and len(t.strip()) > 1 and len(spans) > 1:
                        line_parts.append(f"**{t.strip()}** ")
                    else:
                        line_parts.append(t)
                line_strings.append("".join(line_parts))
            block_text = "\n".join(line_strings)
            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12.0
            blocks.append((x0, y0, x1, y1, block_text, idx, 0, avg_font_size))
        if blocks:
            return blocks
    except Exception:
        pass
    return page.get_text("blocks")


def extract_text_from_pdf(
    path: Path | str,
    remove_headers_footers: bool = True,
    separate_footnotes: bool = True,
    exclude_margin_rubrics: bool = True,
    skip_pages: int = 0,
    take_pages: Optional[int] = None,
) -> str:
    """Extract clean, normalized Arabic text from a PDF with layout awareness and page slicing.

    Parameters
    ----------
    path:
        Path to the PDF file.
    remove_headers_footers:
        Strip running headers and page numbers.
    separate_footnotes:
        Group bottom commentary/footnotes into a dedicated section per page.
    exclude_margin_rubrics:
        Exclude vertical margin tabs and side ribbons (e.g. 'مدخل') from body text.
    skip_pages:
        Number of pages to skip from start (0-indexed).
    take_pages:
        Number of pages to extract.
    """
    if fitz is None:
        raise ImportError(
            "PyMuPDF is required to read PDF files. "
            "Install it with: pip install PyMuPDF"
        )

    doc = fitz.open(str(path))
    total_pages = len(doc)
    start_idx = max(0, skip_pages)
    if start_idx >= total_pages and total_pages > 0:
        doc.close()
        raise ValueError(
            f"skip_pages ({skip_pages}) is beyond the end of the PDF ({total_pages} total pages)."
        )

    end_idx = min(total_pages, start_idx + take_pages) if take_pages is not None else total_pages
    page_texts: List[str] = []

    for page_idx in range(start_idx, end_idx):
        page = doc[page_idx]
        blocks = _extract_blocks_with_styling(page)
        divider_lines = extract_footnote_divider_lines(page)
        rect = page.rect

        body_blocks, footnote_blocks, rubric_texts = classify_page_blocks(
            blocks=blocks,
            page_width=rect.width,
            page_height=rect.height,
            remove_headers_footers=remove_headers_footers,
            separate_footnotes=separate_footnotes,
            exclude_margin_rubrics=exclude_margin_rubrics,
            return_rubrics=True,
            footnote_divider_lines=divider_lines,
        )

        page_parts: List[str] = []
        if rubric_texts and body_blocks:
            rubric_tag = "\n".join(f"<!-- rubric: {r} -->" for r in rubric_texts)
            body_blocks[0] = f"{rubric_tag}\n{body_blocks[0]}"
        elif rubric_texts:
            for r in rubric_texts:
                page_parts.append(f"<!-- rubric: {r} -->")

        if body_blocks:
            page_parts.append(fix_font_ligature_reversals("\n\n".join(body_blocks)))

        if footnote_blocks:
            footnotes_section = (
                "[الهوامش والتعليقات / Page Footnotes]:\n" + fix_font_ligature_reversals("\n".join(footnote_blocks))
            )
            page_parts.append(footnotes_section)

        if page_parts:
            page_texts.append("\n\n".join(page_parts))

    doc.close()
    return "\n\n".join(page_texts)
