"""Unit tests for the Bilingual Alignment & Inline Arabic Typesetting Module."""

import tempfile
import unittest
from pathlib import Path

from translator.bilingual import (
    clean_paragraph_tags,
    format_bilingual_markdown,
    pair_bilingual_paragraphs,
    parse_tagged_paragraphs,
    split_into_paragraphs,
)
from translator.typesetter import (
    compile_markdown_to_pdf,
    markdown_to_typst_content,
)


class TestBilingual(unittest.TestCase):
    def test_split_into_paragraphs(self):
        text = "Første afsnit.\n\nAndet afsnit.\n\n\n\nTredje afsnit."
        paras = split_into_paragraphs(text)
        self.assertEqual(len(paras), 3)
        self.assertEqual(paras[0], "Første afsnit.")
        self.assertEqual(paras[1], "Andet afsnit.")
        self.assertEqual(paras[2], "Tredje afsnit.")

    def test_pair_exact_match(self):
        ar = "الفقرة الأولى.\n\nالفقرة الثانية.\n\nالفقرة الثالثة."
        tr = "Første afsnit.\n\nAndet afsnit.\n\nTredje afsnit."
        pairs = pair_bilingual_paragraphs(ar, tr)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs[0], ("الفقرة الأولى.", "Første afsnit."))
        self.assertEqual(pairs[1], ("الفقرة الثانية.", "Andet afsnit."))
        self.assertEqual(pairs[2], ("الفقرة الثالثة.", "Tredje afsnit."))

    def test_pair_translation_split(self):
        # 2 Arabic paragraphs, but 3 translation paragraphs
        ar = "الفقرة الأولى الطويلة.\n\nالفقرة الثانية."
        tr = "Første del.\n\nAnden del.\n\nTredje afsnit."
        pairs = pair_bilingual_paragraphs(ar, tr)
        self.assertEqual(len(pairs), 2)
        # All translation paragraphs should be preserved
        combined_tr = "\n\n".join(tr for _, tr in pairs)
        self.assertIn("Første del.", combined_tr)
        self.assertIn("Anden del.", combined_tr)
        self.assertIn("Tredje afsnit.", combined_tr)

    def test_pair_translation_merged(self):
        # 3 Arabic paragraphs, but 2 translation paragraphs
        ar = "الفقرة الأولى.\n\nالفقرة الثانية.\n\nالفقرة الثالثة."
        tr = "Første samlede afsnit.\n\nAndet samlede afsnit."
        pairs = pair_bilingual_paragraphs(ar, tr)
        self.assertEqual(len(pairs), 2)
        combined_ar = "\n\n".join(ar_item for ar_item, _ in pairs)
        self.assertIn("الفقرة الأولى.", combined_ar)
        self.assertIn("الفقرة الثانية.", combined_ar)
        self.assertIn("الفقرة الثالثة.", combined_ar)

    def test_tag_arabic_paragraphs(self):
        from translator.bilingual import tag_arabic_paragraphs

        ar = "الفقرة الأولى.\n\nالفقرة الثانية."
        tagged, raw = tag_arabic_paragraphs(ar)
        self.assertEqual(len(raw), 2)
        self.assertIn("[P1]\nالفقرة الأولى.", tagged)
        self.assertIn("[P2]\nالفقرة الثانية.", tagged)

    def test_pair_with_explicit_tags(self):
        ar = "الفقرة الأولى.\n\nالفقرة الثانية."
        # LLM returns tagged translation
        tr = "[P1] Første afsnit oversættelse.\n\n[P2] Andet afsnit oversættelse."
        pairs = pair_bilingual_paragraphs(ar, tr)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0], ("الفقرة الأولى.", "Første afsnit oversættelse."))
        self.assertEqual(pairs[1], ("الفقرة الثانية.", "Andet afsnit oversættelse."))

    def test_format_bilingual_markdown(self):
        ar = "المقدمة في علم المنطق."
        tr = "Indledning til logikkens videnskab."
        formatted = format_bilingual_markdown(ar, tr)
        self.assertIn("> 📜 **[الأصل العربي]**:", formatted)
        self.assertIn("> المقدمة في علم المنطق.", formatted)
        self.assertIn("Indledning til logikkens videnskab.", formatted)

    def test_typesetter_converts_arabic_block(self):
        md = """> 📜 **[الأصل العربي]**:
> هذه المقدمة تستهدف إيضاح الموضوعات التي سيعالجها هذا الجزء.

Denne indledning har til formål at klargøre emnerne.
"""
        typst_content = markdown_to_typst_content(md)
        self.assertTrue("#bilingual_item[" in typst_content or "#arabic_block[" in typst_content)
        self.assertIn("هذه المقدمة تستهدف إيضاح الموضوعات", typst_content)
        self.assertIn("Denne indledning har til formål at klargøre emnerne.", typst_content)

    def test_bilingual_pdf_compilation(self):
        ar = "هذا هو النص العربي الأصلي للمسألة الفقهية الأولى."
        tr = "Dette er den originale arabiske tekst for det første retlige spørgsmål."
        md = format_bilingual_markdown(ar, tr)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_pdf = Path(tmpdir) / "bilingual_test.pdf"
            compile_markdown_to_pdf(
                md_text=md,
                output_pdf_path=out_pdf,
                title="al-Fiqh al-Islāmī",
                subtitle="Tosproget Udgave (Arabisk - Dansk)",
                lang="da",
                domain="fiqh",
                paper_size="b5",
            )
            self.assertTrue(out_pdf.exists())
            self.assertGreater(out_pdf.stat().st_size, 5000)

    def test_parse_tagged_paragraphs_variations(self):
        text = """
[P1]
Første afsnit.

**[P2]**
Andet afsnit.

[P 3]:
Tredje afsnit.

### [P4]
Fjerde afsnit med indre liste:
1. Underpunkt et
2. Underpunkt to

P5:
Femte afsnit.

**P6:**
Sjette afsnit.
"""
        parsed = parse_tagged_paragraphs(text)
        self.assertEqual(len(parsed), 6)
        self.assertEqual(parsed[1], "Første afsnit.")
        self.assertEqual(parsed[2], "Andet afsnit.")
        self.assertEqual(parsed[3], "Tredje afsnit.")
        self.assertIn("Fjerde afsnit med indre liste:", parsed[4])
        self.assertIn("1. Underpunkt et", parsed[4])
        self.assertEqual(parsed[5], "Femte afsnit.")
        self.assertEqual(parsed[6], "Sjette afsnit.")

    def test_verifier_backfill_missing_tags(self):
        from translator.verifier import parse_verification_response

        draft = """[P1]
Første kladde.

[P2]
Anden kladde.

[P3]
Tredje kladde.

[P4]
Fjerde kladde."""

        # Simulate verifier that prematurely truncated after P2
        truncated_verifier_response = """### AUDIT NOTES:
- P1 and P2 checked.

### NOVEL TERMS:
None

### VERIFIED TRANSLATION:
[P1]
Første verificeret.

[P2]
Anden verificeret."""

        notes, verified = parse_verification_response(
            truncated_verifier_response,
            fallback_translation=draft,
        )
        parsed = parse_tagged_paragraphs(verified)
        self.assertEqual(len(parsed), 4)
        self.assertEqual(parsed[1], "Første verificeret.")
        self.assertEqual(parsed[2], "Anden verificeret.")
        # P3 and P4 must be backfilled from draft
        self.assertEqual(parsed[3], "Tredje kladde.")
        self.assertEqual(parsed[4], "Fjerde kladde.")


if __name__ == "__main__":
    unittest.main()
