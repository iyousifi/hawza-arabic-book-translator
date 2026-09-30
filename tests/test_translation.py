"""Unit tests for multi-language glossaries, classical chunking, and translation config."""

import unittest
from pathlib import Path

from translator.config import (
    ModelConfig,
    build_system_prompt,
    get_language_name,
)
from translator.engine import _build_user_message, translate
from translator.file_io import chunk_text, default_output_path, normalize_arabic_text
from translator.glossary import list_available_glossaries, load_glossary, resolve_glossary_path


class TestMultiLanguageGlossary(unittest.TestCase):
    def test_available_languages(self):
        glossaries = list_available_glossaries()
        self.assertIn("en", glossaries)
        self.assertIn("da", glossaries)
        self.assertIn("sw", glossaries)
        self.assertIn("mantiq", glossaries["en"])
        self.assertIn("fiqh", glossaries["en"])
        self.assertIn("mantiq", glossaries["da"])
        self.assertIn("mantiq", glossaries["sw"])

    def test_load_mantiq_english(self):
        glossary = load_glossary(lang="en", domain="mantiq")
        self.assertIn("syllogisms_and_arguments", glossary)
        # Verify Qiyas is Syllogism in Mantiq, NOT analogical reasoning
        self.assertIn("syllogism", glossary["syllogisms_and_arguments"]["القياس"].lower())
        self.assertIn("conception", glossary["foundational_divisions"]["التصور"].lower())

    def test_load_mantiq_danish(self):
        glossary = load_glossary(lang="da", domain="mantiq")
        self.assertIn("syllogisms_and_arguments", glossary)
        self.assertIn("syllogisme", glossary["syllogisms_and_arguments"]["القياس"].lower())
        self.assertIn("fred være med ham", glossary["honorifics"]["عليه السلام"].lower())

    def test_load_mantiq_swahili(self):
        glossary = load_glossary(lang="sw", domain="mantiq")
        self.assertIn("syllogisms_and_arguments", glossary)
        self.assertIn("silojia", glossary["syllogisms_and_arguments"]["القياس"].lower())
        self.assertIn("amani iwe juu yake", glossary["honorifics"]["عليه السلام"].lower())


class TestPromptAndConfig(unittest.TestCase):
    def test_language_names(self):
        self.assertEqual(get_language_name("en"), "English")
        self.assertEqual(get_language_name("da"), "Danish")
        self.assertEqual(get_language_name("sw"), "Swahili")

    def test_system_prompt_includes_language_and_logic_rules(self):
        prompt_da = build_system_prompt(target_lang="da", domain="mantiq")
        self.assertIn("Danish", prompt_da)
        self.assertIn("Classical Scholastic Arabic", prompt_da)
        self.assertIn("Pronoun Resolution", prompt_da)
        self.assertIn("Diacritical", prompt_da)

        prompt_sw = build_system_prompt(target_lang="sw", domain="mantiq")
        self.assertIn("Swahili", prompt_sw)

    def test_user_message_with_context(self):
        msg_without = _build_user_message("نص عربي", target_lang="da")
        self.assertIn("Danish", msg_without)
        self.assertNotIn("Preceding", msg_without)

        msg_with = _build_user_message(
            "نص عربي",
            target_lang="da",
            prev_target_context="Forrige oversættelsesstykke",
            prev_arabic_context="السياق العربي السابق",
        )
        self.assertIn("Preceding Arabic Context", msg_with)
        self.assertIn("السياق العربي السابق", msg_with)
        self.assertIn("Preceding Danish Translation", msg_with)
        self.assertIn("Forrige oversættelsesstykke", msg_with)


class TestFileIOAndChunking(unittest.TestCase):
    def test_default_output_path(self):
        self.assertEqual(default_output_path("book.pdf", lang="en"), "book_translated.md")
        self.assertEqual(default_output_path("book.pdf", lang="da"), "book_translated_da.md")
        self.assertEqual(default_output_path("book.pdf", lang="sw"), "book_translated_sw.md")

    def test_arabic_normalization(self):
        sample = "الـتـصــور والـتـصـديـق\u200b"
        cleaned = normalize_arabic_text(sample)
        self.assertEqual(cleaned, "التصور والتصديق")

    def test_classical_discourse_chunking(self):
        long_para = (
            "المبحث الأول في تعريف علم المنطق وبيان الحاجة إليه "
            "أما بعد فإن الإنسان مفطور على التفكير "
            "وحينئذ يعرض لفكره الخطأ والانحراف "
            "ومن هنا مست الحاجة إلى قانون يعصم الذهن عن الخطأ "
            "فإن قيل إن العقل كاف في درك الحقائق "
            "قلنا إن العقل يقع في الخطأ وجدانا فلا بد له من ميزان ضابط"
        )
        chunks = chunk_text(long_para, max_tokens=15)
        self.assertTrue(len(chunks) > 1)

    def test_chunk_classical_arabic_with_overlap(self):
        from translator.chunker import chunk_classical_arabic

        scholastic_text = (
            "المقصد الأول في مباحث الألفاظ\n\n"
            "الفصل الأول في دلالة اللفظ\n"
            "إذا أطلق اللفظ وأريد به المعنى فإن السامع ينتقل ذهنه من سماع اللفظ إلى تصور المعنى. "
            "وحيث إن هذا الانتقال سببه الوضع، سميت الدلالة وضعية.\n\n"
            "فإن قيل إن الدلالة قد تحصل بغير وضع كالطبيعية والعقلية.\n"
            "قلنا نعم، ولكن غرض المنطقي مقصور على الدلالة الوضعية اللفظية، "
            "لأنها هي العمدة في التفاهم ونقل الأفكار.\n\n"
            "تنبيه: اعلم أن الألفاظ هي قوالب المعاني، والمقصود بالذات هو المعنى دون اللفظ. "
            "ومن هنا كان بحث المنطقي عن الألفاظ بحثا بالعرض لا بالذات."
        )

        chunks = chunk_classical_arabic(scholastic_text, max_tokens=30, overlap_tokens=10)
        self.assertTrue(len(chunks) >= 2)

        # Chunk 0 has no preceding context
        self.assertEqual(chunks[0].prev_arabic_context, "")

        # Chunk 1 and later must contain antecedent context from the preceding chunk
        self.assertTrue(len(chunks[1].prev_arabic_context) > 0)
        # Ensure antecedent context matches text from chunk 0
        self.assertIn(chunks[1].prev_arabic_context, chunks[0].text)


class TestSmartNormalizer(unittest.TestCase):
    def test_presentation_forms_conversion(self):
        # \uFEFB is isolated Arabic ligature Lam-Alif, \uFDF2 is Arabic ligature Allah
        raw = "\uFEFB \uFDF2 \uFED7\uFEE2"
        normalized = normalize_arabic_text(raw)
        self.assertEqual(normalized, "لا الله قم")

    def test_bidi_and_tatweel_cleaning(self):
        # Arabic with LRM \u200E, RLM \u200F, zero-width space \u200B, and Tatweel \u0640
        raw = "\u200Eالـ\u0640كـ\u0640لـ\u0640ي\u200F \u200Bوالـ\u0640جـ\u0640زئـ\u0640ي"
        normalized = normalize_arabic_text(raw)
        self.assertEqual(normalized, "الكلي والجزئي")

    def test_classify_page_blocks(self):
        from translator.normalizer import classify_page_blocks

        # Simulate a page of height 1000, width 600
        # (x0, y0, x1, y1, text, block_no, block_type)
        blocks = [
            (50.0, 20.0, 550.0, 50.0, "كتاب المنطق - الباب الأول", 0, 0),  # Header at top (<7.5%)
            (50.0, 100.0, 550.0, 200.0, "القياس قول مؤلف من قضايا متى سلمت لزم عنها لذاتها قول آخر.", 1, 0),  # Body
            (50.0, 220.0, 550.0, 320.0, "وهذا القول الآخر يسمى بالنتيجة.", 2, 0),  # Body
            (50.0, 800.0, 550.0, 850.0, "(١) هذا تعريف القياس عند الشيخ الرئيس في الشفاء.", 3, 0),  # Footnote at bottom
            (250.0, 960.0, 350.0, 980.0, "٤٥", 4, 0),  # Page number at bottom (>92.5%)
        ]

        body, footnotes = classify_page_blocks(
            blocks=blocks,
            page_width=600.0,
            page_height=1000.0,
            remove_headers_footers=True,
            separate_footnotes=True,
        )

        # Header and page number should be removed
        self.assertEqual(len(body), 2)
        self.assertIn("القياس قول مؤلف", body[0])
        self.assertIn("وهذا القول الآخر", body[1])

        # Footnote separated
        self.assertEqual(len(footnotes), 1)
        self.assertIn("هذا تعريف القياس", footnotes[0])

    def test_extract_text_from_pdf_end_to_end(self):
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        from translator.normalizer import extract_text_from_pdf
        import tempfile

        # Generate a synthetic PDF with header, body, and footnote
        doc = fitz.open()
        page = doc.new_page(width=600, height=1000)
        font_kw = (
            {"fontfile": "C:/Windows/Fonts/arial.ttf", "fontname": "arial"}
            if Path("C:/Windows/Fonts/arial.ttf").exists()
            else {}
        )

        # Top header (< 7.5% height)
        page.insert_text((50, 40), "ترجمة المنطق", fontsize=10, **font_kw)
        # Body text
        page.insert_text((50, 250), "العلم إما تصور وإما تصديق", fontsize=14, **font_kw)
        # Footnote (> 65% height with marker)
        page.insert_text((50, 750), "(١) التصور إدراك الساذج", fontsize=9, **font_kw)
        # Solitary page number in footer (> 92.5%)
        page.insert_text((290, 960), "١٢", fontsize=10, **font_kw)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            doc.save(str(tmp_path))
            doc.close()

            extracted = extract_text_from_pdf(tmp_path)
            # Body text must be present
            self.assertIn("العلم إما تصور وإما تصديق", extracted)
            # Footnotes section must be cleanly separated
            self.assertIn("[الهوامش والتعليقات / Page Footnotes]:", extracted)
            self.assertIn("التصور إدراك الساذج", extracted)
            # Running header and solitary page number should be stripped
            self.assertNotIn("ترجمة المنطق", extracted)
            self.assertNotIn("١٢", extracted)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_margin_rubric_exclusion(self):
        from translator.normalizer import classify_page_blocks

        # Simulated page: width 600, height 800
        # Block 1: Body text in main column
        # Block 2: Margin rubric "مدخل" at x0=545 (545/600 = 0.908 > 0.86), w=25, h=80 (>1.8*w)
        blocks = [
            (50.0, 100.0, 500.0, 200.0, "النص الرئيسي للدرس الأول.\n", 0, 0),
            (545.0, 150.0, 570.0, 230.0, "مــــــــدخــــــــل\n", 1, 0),
        ]

        # With exclude_margin_rubrics=True (default)
        body_excl, _ = classify_page_blocks(blocks, page_width=600.0, page_height=800.0, exclude_margin_rubrics=True)
        self.assertEqual(len(body_excl), 1)
        self.assertIn("النص الرئيسي", body_excl[0])
        self.assertTrue(not any("مدخل" in b for b in body_excl))

        # With exclude_margin_rubrics=False
        body_kept, _ = classify_page_blocks(blocks, page_width=600.0, page_height=800.0, exclude_margin_rubrics=False)
        self.assertEqual(len(body_kept), 2)

    def test_typesetter_palette_styling(self):
        from translator.typesetter import build_typst_document

        doc = build_typst_document("= Overskrift\n1. Første punkt", title="Test Bog", lang="da")
        # Crimson red for lists
        self.assertIn("#d2232a", doc)
        # Royal blue for subheadings
        self.assertIn("#24408f", doc)
        # Deep navy for main headings/title
        self.assertIn("#2e3092", doc)
        # Charcoal for body text
        self.assertIn("#231f20", doc)
        # Parchment background and amber border for Arabic blocks
        self.assertIn("#faf8f5", doc)
        self.assertIn("#b08968", doc)

    def test_merge_fragmented_blocks(self):
        from translator.normalizer import merge_fragmented_blocks

        blocks = [
            ":الدرس الثاني",
            "تطور البحث المنطقي",
            "يستهدف هذا التمهيد بيان الموضوعات التي سيتناولها هذا القسم،",
            "كما يستهدف بيان المنهجية التي يتبعها الكتاب.",
            "مدخل:",
            "في حديثنا عن تطور المنطق نلقي نظرة على الأدوار.",
        ]
        merged = merge_fragmented_blocks(blocks)
        # Heading :الدرس الثاني should be separate
        self.assertEqual(merged[0], ":الدرس الثاني")
        # Broken lines should be merged
        self.assertIn("يستهدف هذا التمهيد بيان الموضوعات التي سيتناولها هذا القسم، كما يستهدف بيان المنهجية التي يتبعها الكتاب.", merged)



class TestDryRunTranslation(unittest.TestCase):
    def test_dry_run_multi_lang(self):
        cfg_da = ModelConfig(target_lang="da", domain="mantiq")
        result_da = translate("التصور والتصديق من مباحث علم المنطق.", cfg=cfg_da, dry_run=True)
        self.assertIn("[DRY RUN - DANISH (mantiq)]", result_da)

        cfg_sw = ModelConfig(target_lang="sw", domain="mantiq")
        result_sw = translate("التصور والتصديق من مباحث علم المنطق.", cfg=cfg_sw, dry_run=True)
        self.assertIn("[DRY RUN - SWAHILI (mantiq)]", result_sw)

    def test_dry_run_with_verification(self):
        cfg_verified = ModelConfig(target_lang="da", domain="mantiq", verify=True)
        result = translate("القياس حجة مؤلفة من قضايا.", cfg=cfg_verified, dry_run=True)
        self.assertIn("- VERIFIED", result)
        self.assertIn("[DRY RUN - DANISH (mantiq) - VERIFIED]", result)


class TestTwoPassVerification(unittest.TestCase):
    def test_parse_verification_response_structured(self):
        from translator.verifier import parse_verification_response

        raw_response = (
            "### AUDIT NOTES:\n"
            "- Corrected quantifier in minor premise from 'some' to 'every'.\n"
            "- Standardized Qiyas to 'syllogism (qiyas)'.\n\n"
            "### VERIFIED TRANSLATION:\n"
            "Every human is an animal, and every animal is a mortal."
        )

        notes, verified = parse_verification_response(raw_response)
        self.assertIn("Corrected quantifier", notes)
        self.assertIn("Standardized Qiyas", notes)
        self.assertEqual(verified, "Every human is an animal, and every animal is a mortal.")

    def test_parse_verification_response_fallback(self):
        from translator.verifier import parse_verification_response

        raw = "Direct translation without markdown headers."
        notes, verified = parse_verification_response(raw)
        self.assertEqual(verified, raw)

    def test_verifier_system_prompt_criteria(self):
        from translator.config import build_verifier_system_prompt

        prompt_da = build_verifier_system_prompt(target_lang="da", domain="mantiq")
        self.assertIn("Danish", prompt_da)
        self.assertIn("Quantifiers & Modalities", prompt_da)
        self.assertIn("Negations & Polarity", prompt_da)
        self.assertIn("Syllogistic & Premise Integrity", prompt_da)
        self.assertIn("smooth hallucinations", prompt_da)

        prompt_sw = build_verifier_system_prompt(target_lang="sw", domain="mantiq")
        self.assertIn("Swahili", prompt_sw)


class TestPageAndChapterSlicing(unittest.TestCase):
    def test_parse_page_range(self):
        from translator.slicing import format_page_tag, parse_page_range

        self.assertEqual(parse_page_range("10-25"), (9, 16))
        self.assertEqual(parse_page_range("10:25"), (9, 16))
        self.assertEqual(parse_page_range("10:"), (9, None))
        self.assertEqual(parse_page_range(":20"), (0, 20))
        self.assertEqual(parse_page_range("5"), (4, 1))
        self.assertEqual(parse_page_range(None, skip=10, take=5), (10, 5))
        self.assertEqual(parse_page_range(None), (0, None))

        with self.assertRaises(ValueError):
            parse_page_range("25-10")

        with self.assertRaises(ValueError):
            parse_page_range("invalid_range")

        self.assertEqual(format_page_tag(9, 16), "p10-25")
        self.assertEqual(format_page_tag(4, 1), "p5")
        self.assertEqual(format_page_tag(9, None), "p10-end")
        self.assertEqual(format_page_tag(0, None), "")

    def test_detect_and_extract_chapters(self):
        from translator.slicing import detect_chapters, extract_chapter

        text = (
            "المقدمة: في تعريف علم المنطق وبيان الحاجة إليه.\n"
            "المنطق آلة قانونية تعصم مراعاتها الذهن عن الخطأ في الفكر.\n\n"
            "المقصد الأول: في مباحث الألفاظ ودلالتها.\n"
            "الدلالة تنقسم إلى عقلية وطبيعية ووضعية.\n\n"
            "المقصد الثاني: في مباحث الكلي والجزئي.\n"
            "الكلي ما لا يمتنع فرض صدقه على كثيرين.\n\n"
            "الخاتمة: في المغالطات والسفسطة.\n"
            "المغالطة حجة فاسدة تعطي نتيجة غير صحيحة."
        )

        chapters = detect_chapters(text)
        self.assertEqual(len(chapters), 4)
        self.assertIn("المقدمة", chapters[0].title)
        self.assertIn("المقصد الأول", chapters[1].title)
        self.assertIn("المقصد الثاني", chapters[2].title)
        self.assertIn("الخاتمة", chapters[3].title)

        # Extract by index (1-indexed)
        chap1_text, info1 = extract_chapter(text, 1)
        self.assertIn("تعريف علم المنطق", chap1_text)
        self.assertNotIn("المقصد الأول", chap1_text)
        self.assertEqual(info1.index, 1)

        # Extract by title query
        chap2_text, info2 = extract_chapter(text, "مباحث الألفاظ")
        self.assertIn("الدلالة تنقسم", chap2_text)
        self.assertEqual(info2.index, 2)

        # Extract non-existent chapter
        with self.assertRaises(ValueError):
            extract_chapter(text, 99)

        with self.assertRaises(ValueError):
            extract_chapter(text, "فصل غير موجود")

    def test_output_path_with_slice_tags(self):
        from translator.file_io import default_output_path

        path_page = default_output_path("kitab.pdf", lang="da", slice_tag="p10-25")
        self.assertEqual(path_page, "kitab_p10-25_translated_da.md")

        path_chap = default_output_path("kitab.pdf", lang="en", slice_tag="ch2")
        self.assertEqual(path_chap, "kitab_ch2_translated.md")

        path_both = default_output_path("kitab.pdf", lang="sw", slice_tag="ch1_p1-10")
        self.assertEqual(path_both, "kitab_ch1_p1-10_translated_sw.md")


class TestDynamicTerminologyMemory(unittest.TestCase):
    def test_record_term_normalization_and_dedup(self):
        from translator.memory import TerminologyMemory

        mem = TerminologyMemory()
        # Add vocalized term
        added = mem.record_term("العِلْمُ الحُضُورِيُّ", "knowledge by presence", source_chunk=1)
        self.assertTrue(added)

        # Lookup with unvocalized string
        self.assertEqual(mem.get_term("العلم الحضوري"), "knowledge by presence")
        self.assertTrue(mem.has_term("العلم الحضوري"))

        # Duplicate without overwrite should return False and keep original
        added_dup = mem.record_term("العلم الحضوري", "immanent knowing", overwrite=False)
        self.assertFalse(added_dup)
        self.assertEqual(mem.get_term("العلم الحضوري"), "knowledge by presence")

        # Overwrite should succeed
        mem.record_term("العلم الحضوري", "immanent knowing", overwrite=True)
        self.assertEqual(mem.get_term("العلم الحضوري"), "immanent knowing")

        # Honorifics should be ignored
        added_hon = mem.record_term("صلى الله عليه وآله وسلم", "peace be upon him")
        self.assertFalse(added_hon)

    def test_extract_terms_from_audit(self):
        from translator.memory import TerminologyMemory

        mem = TerminologyMemory()
        audit_text = (
            "### AUDIT NOTES:\n"
            "- Corrected quantifier in major premise.\n\n"
            "### NOVEL TERMS:\n"
            "- العلم الحصولي: acquired knowledge\n"
            "* الوجوب الغيري = extrinsic necessity\n"
            "- Simple conception: التصور الساذج\n"
        )
        extracted = mem.extract_from_audit(audit_text)
        self.assertEqual(len(extracted), 3)
        self.assertIn("العلم الحصولي", extracted)
        self.assertEqual(extracted["العلم الحصولي"], "acquired knowledge")
        self.assertIn("الوجوب الغيري", extracted)
        self.assertEqual(extracted["الوجوب الغيري"], "extrinsic necessity")
        self.assertIn("التصور الساذج", extracted)
        self.assertEqual(extracted["التصور الساذج"], "Simple conception")

    def test_extract_terms_from_text(self):
        from translator.memory import TerminologyMemory

        mem = TerminologyMemory()
        text = (
            "The author establishes knowledge by presence (العلم الحضوري) as foundational. "
            "Furthermore, [i.e., القياس الاستثنائي: conditional syllogism] is divided into two modes."
        )
        extracted = mem.extract_from_text(text)
        self.assertIn("العلم الحضوري", extracted)
        self.assertEqual(extracted["العلم الحضوري"], "knowledge by presence")
        self.assertIn("القياس الاستثنائي", extracted)
        self.assertEqual(extracted["القياس الاستثنائي"], "conditional syllogism")

    def test_format_for_prompt_and_static_filtering(self):
        from translator.memory import TerminologyMemory

        mem = TerminologyMemory()
        # Seed static term (source_chunk=-1)
        mem.record_term("القياس", "syllogism", source_chunk=-1)
        # Add dynamic term (source_chunk=1)
        mem.record_term("العلم الحضوري", "knowledge by presence", source_chunk=1)

        # By default, prompt formatting should only include novel dynamic terms
        prompt_block = mem.format_for_prompt()
        self.assertIn("العلم الحضوري", prompt_block)
        self.assertIn("knowledge by presence", prompt_block)
        self.assertNotIn("القياس: syllogism", prompt_block)

    def test_save_and_load_json(self):
        import tempfile
        from pathlib import Path
        from translator.memory import TerminologyMemory

        mem = TerminologyMemory()
        mem.record_term("الوجوب بالذات", "essential necessity", source_chunk=1)
        mem.record_term("الإمكان الخاص", "specific contingency", source_chunk=2)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            mem.save_json(tmp_path)
            self.assertTrue(tmp_path.exists())

            # Load into new memory bank
            new_mem = TerminologyMemory.from_file(tmp_path)
            self.assertEqual(len(new_mem), 2)
            self.assertEqual(new_mem.get_term("الوجوب بالذات"), "essential necessity")
            self.assertEqual(new_mem.get_term("الإمكان الخاص"), "specific contingency")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_parse_verification_response_with_novel_terms(self):
        from translator.verifier import parse_verification_response

        raw_response = (
            "### AUDIT NOTES:\n"
            "- Polarity was preserved.\n\n"
            "### NOVEL TERMS:\n"
            "- العلم الحضوري: kendskab ved nærvær\n\n"
            "### VERIFIED TRANSLATION:\n"
            "Videnskab er enten forestilling eller bekræftelse."
        )

        # Backward-compatible 2-element unpack
        notes, verified = parse_verification_response(raw_response)
        self.assertEqual(verified, "Videnskab er enten forestilling eller bekræftelse.")
        self.assertIn("Polarity was preserved", notes)

        # 3-element unpack with extract_terms=True
        notes, verified, terms = parse_verification_response(raw_response, extract_terms=True)
        self.assertIn("العلم الحضوري", terms)
        self.assertEqual(terms["العلم الحضوري"], "kendskab ved nærvær")

    def test_engine_dry_run_with_session_glossary(self):
        import tempfile
        from pathlib import Path
        from translator.config import ModelConfig
        from translator.engine import translate

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            cfg = ModelConfig(target_lang="da", domain="mantiq")
            result = translate(
                "العلم إما تصور وإما تصديق.",
                cfg=cfg,
                save_session_glossary=str(tmp_path),
                dry_run=True,
            )
            self.assertIn("[DRY RUN - DANISH", result)
            self.assertTrue(tmp_path.exists())
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_build_user_message_with_session_terms(self):
        from translator.engine import _build_user_message

        msg = _build_user_message(
            chunk="النص العربي",
            target_lang="da",
            prev_target_context="Tidligere oversættelse",
            prev_arabic_context="السياق السابق",
            session_terms_block="[Dynamic Session Terminology]:\n- العلم الحضوري: kendskab ved nærvær",
        )
        self.assertIn("Dynamic Session Terminology", msg)
        self.assertIn("kendskab ved nærvær", msg)
        self.assertIn("Tidligere oversættelse", msg)
        self.assertIn("السياق السابق", msg)
        self.assertIn("النص العربي", msg)


class TestVerificationAuditReport(unittest.TestCase):
    def test_default_audit_path(self):
        from translator.file_io import default_audit_path

        p1 = default_audit_path("kitab_p10-25_translated_da.md")
        self.assertEqual(p1, "kitab_p10-25_translated_da_audit.md")

        p2 = default_audit_path("output.md")
        self.assertEqual(p2, "output_audit.md")

    def test_generate_audit_report(self):
        from translator.engine import AuditReportEntry, generate_audit_report

        entries = [
            AuditReportEntry(
                chunk_index=1,
                total_chunks=2,
                has_corrections=True,
                audit_notes="Corrected quantifier from some to all.",
                novel_terms={"العلم الحضوري": "kendskab ved nærvær"},
            ),
            AuditReportEntry(
                chunk_index=2,
                total_chunks=2,
                has_corrections=False,
                audit_notes="No critical deviations detected.",
            ),
        ]

        report = generate_audit_report(
            entries=entries,
            target_lang="da",
            domain="mantiq",
            provider="gemini",
            model="gemini-3.8-flash",
            verifier_model="gemini-3.8-flash",
        )

        self.assertIn("# 🔍 Scholastic Verification & Audit Report", report)
        self.assertIn("Danish (`da`)", report)
        self.assertIn("Chunks with Corrections**: 1 / 2", report)
        self.assertIn("Corrected quantifier from some to all", report)
        self.assertIn("العلم الحضوري", report)
        self.assertIn("kendskab ved nærvær", report)

    def test_dry_run_generates_audit_report(self):
        import tempfile
        from pathlib import Path
        from translator.config import ModelConfig
        from translator.engine import translate

        with tempfile.NamedTemporaryFile(suffix="_audit.md", delete=False) as tmp:
            tmp_audit = Path(tmp.name)

        try:
            cfg = ModelConfig(target_lang="da", domain="mantiq", verify=True)
            result = translate(
                "العلم إما تصور وإما تصديق.",
                cfg=cfg,
                audit_report_path=str(tmp_audit),
                dry_run=True,
            )
            self.assertIn("[DRY RUN", result)
            self.assertTrue(tmp_audit.exists())
            content = tmp_audit.read_text(encoding="utf-8")
            self.assertIn("Scholastic Verification & Audit Report", content)
        finally:
            if tmp_audit.exists():
                tmp_audit.unlink()


if __name__ == "__main__":
    unittest.main()




