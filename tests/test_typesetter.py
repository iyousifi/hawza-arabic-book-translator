"""Unit tests for the Academic Typesetter module (Markdown -> Typst -> PDF)."""

import os
import tempfile
import unittest
from pathlib import Path

from translator.typesetter import (
    build_typst_document,
    compile_markdown_to_pdf,
    is_typst_available,
    markdown_to_typst_content,
    normalize_paper_size,
)


class TestTypesetter(unittest.TestCase):
    def test_typst_availability(self):
        self.assertTrue(is_typst_available(), "Typst engine should be installed and available")

    def test_paper_size_normalization(self):
        self.assertEqual(normalize_paper_size("b5"), "iso-b5")
        self.assertEqual(normalize_paper_size("iso-b5"), "iso-b5")
        self.assertEqual(normalize_paper_size("a4"), "a4")
        self.assertEqual(normalize_paper_size("a5"), "a5")
        self.assertEqual(normalize_paper_size("letter"), "us-letter")
        self.assertEqual(normalize_paper_size("unknown"), "iso-b5")

    def test_markdown_formatting_conversion(self):
        md = """# Lektion 1

Dette er en **vigtig** introduktion med *kursiv* tekst og $100 og user@example.com.

> Qur'an 2:255: «Gud er den Levende»

## Undersøgelse

- Første punkt
- Andet punkt
1. Nummeret punkt
"""
        converted = markdown_to_typst_content(md)
        self.assertIn("= Lektion 1", converted)
        self.assertIn("== Undersøgelse", converted)
        self.assertIn("*vigtig*", converted)
        self.assertIn("_kursiv_", converted)
        self.assertIn(r"\$100", converted)
        self.assertIn(r"\@example", converted)
        self.assertIn("#quran_callout[", converted)
        self.assertIn("- Første punkt", converted)
        self.assertIn("+ Nummeret punkt", converted)

    def test_footnote_conversion(self):
        md = """Dette er en sætning med en reference[^1] til en vigtig kilde.
Og en anden sætning[^2].

[^1]: Se *al-Kāfī*, bind 1, s. 25.
[^2]: Se *Tahdhīb al-Aḥkām*, bind 2, s. 10.
"""
        converted = markdown_to_typst_content(md)
        self.assertIn("#footnote[Se _al-Kāfī_, bind 1, s. 25.]", converted)
        self.assertIn("#footnote[Se _Tahdhīb al-Aḥkām_, bind 2, s. 10.]", converted)
        # Footnote definitions should be stripped from body text
        self.assertNotIn("[^1]:", converted)
        self.assertNotIn("[^2]:", converted)

    def test_compile_markdown_to_pdf(self):
        sample_md = """# Den logiske forsknings udvikling

Denne indledning redegør for metodologien i undersøgelsen af den logiske tænkning (*fikr*)[^1].

> *«Forstandens første princip er distinktionen mellem det essentielle og det accidentielle.»*

## Den græske periode

Sokrates indførte dialogens metode, mens Aristoteles systematiserede den formelle logik.

[^1]: Kilde: *al-Manṭiq al-Islāmī*, lektion 2.
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_pdf = Path(tmpdir) / "test_output.pdf"
            result_path = compile_markdown_to_pdf(
                md_text=sample_md,
                output_pdf_path=out_pdf,
                title="Den Logiske Forsknings Udvikling",
                subtitle="Hawza Pensum Oversættelse",
                lang="da",
                domain="mantiq",
                paper_size="b5",
            )
            self.assertTrue(result_path.exists())
            self.assertGreater(result_path.stat().st_size, 1000)


    def test_compile_with_unclosed_labels_and_symbols(self):
        sample_md = """# Lektion om Jurisprudens

> 📜 **[الأصل العربي]**:
> هو قول الهل تعاىل: <اĔقْرَ

Allahs ord: «Læs!»

[Fodnote:]* Indsigt i religionen er obligatorisk.

Column 1: `ذلِكَ . وعَنْ زُرَارَةَ بْنِ أَعْيَنَ

> 📜 **[الأصل العربي]**:
> الَّȈذ ِıي عَلَّمَ øبِالْēق َلَمِ * عَلَّمَ الŅإ ِōنْسَاĩن َ
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_pdf = Path(tmpdir) / "test_unclosed_sanitized.pdf"
            result_path = compile_markdown_to_pdf(
                md_text=sample_md,
                output_pdf_path=out_pdf,
                title="Fiqh 3 Test",
                lang="da",
                domain="fiqh",
                paper_size="b5",
            )
            self.assertTrue(result_path.exists())
            self.assertGreater(result_path.stat().st_size, 1000)

    def test_arabic_font_selection(self):
        doc_amiri = build_typst_document("body", arabic_font="amiri")
        self.assertIn('"Amiri", "Dubai"', doc_amiri)

        doc_dubai = build_typst_document("body", arabic_font="dubai")
        self.assertIn('"Dubai", "Amiri"', doc_dubai)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_pdf = Path(tmpdir) / "test_amiri.pdf"
            sample_md = "> اقْرَأْ بِاسْمِ رَبِّكَ الَّذِي خَلَقَ\n\nLæs i din Herres navn."
            res = compile_markdown_to_pdf(
                md_text=sample_md,
                output_pdf_path=out_pdf,
                arabic_font="amiri",
            )
            self.assertTrue(res.exists())

    def test_strip_harakat_in_typesetting(self):
        md = "> 📜 **[الأصل العربي]**:\n> اقْرَأْ بِاسْمِ رَبِّكَ الَّذِي خَلَقَ\n\nLæs i din Herres navn."
        converted_no_vowels = markdown_to_typst_content(md, strip_harakat=True, keep_shaddah=False)
        self.assertIn("اقرأ باسم ربك الذي خلق", converted_no_vowels)
        self.assertNotIn("اقْرَأْ", converted_no_vowels)

        converted_keep_shaddah = markdown_to_typst_content(md, strip_harakat=True, keep_shaddah=True)
        self.assertIn("اقرأ باسم ربّك الّذي خلق", converted_keep_shaddah)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_pdf = Path(tmpdir) / "test_unvocalized.pdf"
            res = compile_markdown_to_pdf(
                md_text=md,
                output_pdf_path=out_pdf,
                strip_harakat=True,
                arabic_font="amiri",
            )
            self.assertTrue(res.exists())
            self.assertGreater(res.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
