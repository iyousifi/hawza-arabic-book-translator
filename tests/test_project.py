"""Unit tests for Hawza Translation Project Management."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from translator.project import (
    ProjectSlice,
    ProjectState,
    continue_project,
    init_project,
    inspect_existing_bilingual_md,
    offset_markdown_footnotes,
    print_project_status,
)


class TestProjectManagement(unittest.TestCase):
    def test_offset_markdown_footnotes(self):
        sample_md = (
            "Dette er en tekst med en reference[^1] og en anden reference[^2].\n"
            "Og en tredje reference[^3].\n\n"
            "[^1]: Første fodnote.\n"
            "[^2]: Anden fodnote.\n"
            "[^3]: Tredje fodnote.\n"
        )
        updated, max_fn = offset_markdown_footnotes(sample_md, offset=14)
        self.assertEqual(max_fn, 17)
        self.assertIn("[^15]", updated)
        self.assertIn("[^16]", updated)
        self.assertIn("[^17]", updated)
        self.assertIn("[^15]: Første fodnote.", updated)
        self.assertIn("[^16]: Anden fodnote.", updated)
        self.assertIn("[^17]: Tredje fodnote.", updated)
        self.assertNotIn("[^1]:", updated)

    def test_offset_markdown_footnotes_zero_offset(self):
        sample_md = "Tekst[^5].\n\n[^5]: Note."
        updated, max_fn = offset_markdown_footnotes(sample_md, offset=0)
        self.assertEqual(max_fn, 5)
        self.assertEqual(updated, sample_md)

    def test_inspect_existing_bilingual_md(self):
        md_text = (
            "# Lektion 1\n\n"
            "> المقدمة في بيان الأصول.\n\n"
            "Indledning til redegørelse for principperne.[^1]\n\n"
            "> وذهب الشيخ إلى أن المسألة أصولية.\n\n"
            "Sheikhen mente, at spørgsmålet er af fundamental karakter.\n\n"
            "[^1]: Se al-Kafi, bind 1.\n"
        )
        inspection = inspect_existing_bilingual_md(md_text)
        self.assertEqual(inspection["max_footnote_index"], 1)
        self.assertIn("وذهب الشيخ إلى أن المسألة أصولية.", inspection["trailing_arabic"])
        self.assertIn("Sheikhen mente, at spørgsmålet er af fundamental karakter.", inspection["trailing_target"])

    def test_init_project_scratch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src_file = tmp / "sample_book.txt"
            src_file.write_text("نص الكتاب العربي.", encoding="utf-8")

            proj_file = tmp / "my_book.project.json"
            state = init_project(
                project_file=proj_file,
                source_file=src_file,
                domain="fiqh",
                target_lang="da",
                book_title="Fiqh Bog",
            )

            self.assertTrue(proj_file.is_file())
            self.assertEqual(state.project_name, "my_book")
            self.assertEqual(state.domain, "fiqh")
            self.assertEqual(state.target_lang, "da")
            self.assertEqual(state.next_page, 1)
            self.assertEqual(len(state.slices), 0)

            # Test loading from file
            loaded = ProjectState.load(proj_file)
            self.assertEqual(loaded.project_name, "my_book")
            self.assertEqual(loaded.book_title, "Fiqh Bog")

    def test_init_project_adopt_existing_translation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src_file = tmp / "Fiqh3.txt"
            src_file.write_text("المحتوى الكامل لكتاب الفقه.", encoding="utf-8")

            existing_md = tmp / "Fiqh3_p11-23.md"
            existing_md.write_text(
                "> مسألة 1: الاجتهاد والتقليد واجبان.\n\n"
                "Spørgsmål 1: Ijtihad og taqlid er obligatoriske.[^2]\n\n"
                "[^2]: Se Masail.\n",
                encoding="utf-8",
            )

            proj_file = tmp / "Fiqh3.project.json"
            state = init_project(
                project_file=proj_file,
                source_file=src_file,
                domain="fiqh",
                target_lang="da",
                adopt_md=existing_md,
                adopt_pages="11-23",
            )

            self.assertEqual(state.next_page, 24)
            self.assertEqual(len(state.slices), 1)
            self.assertEqual(state.slices[0].start_page, 11)
            self.assertEqual(state.slices[0].end_page, 23)
            self.assertEqual(state.max_footnote_index, 2)
            self.assertIn("مسألة 1", state.trailing_arabic)
            self.assertTrue(Path(state.master_md).is_file())

    def test_continue_project_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src_file = tmp / "Fiqh3.txt"
            # 5 pages separated by \f
            src_file.write_text(
                "الفقرة الأولى في بيان الحكم.\f"
                "الفقرة الثانية في تفصيل المسألة.\f"
                "الفقرة الثالثة في الدليل العقلي.\f"
                "الفقرة الرابعة في الدليل النقلي.\f"
                "الفقرة الخامسة في الخاتمة.\n",
                encoding="utf-8",
            )

            existing_md = tmp / "Fiqh3_p1-2.md"
            existing_md.write_text(
                "> الفقرة التمهيدية.\n\n"
                "Indledende afsnit oversættelse.[^1]\n\n"
                "[^1]: Note 1.\n",
                encoding="utf-8",
            )

            proj_file = tmp / "Fiqh3.project.json"
            init_project(
                project_file=proj_file,
                source_file=src_file,
                domain="fiqh",
                target_lang="da",
                adopt_md=existing_md,
                adopt_pages="1-2",
            )

            # Continue next slice in dry run mode
            updated_state = continue_project(
                project_file=proj_file,
                pages_str="3-4",
                dry_run=True,
                compile_pdf=False,
            )

            self.assertEqual(updated_state.next_page, 5)
            self.assertEqual(len(updated_state.slices), 2)
            self.assertEqual(updated_state.slices[1].start_page, 3)
            self.assertEqual(updated_state.slices[1].end_page, 4)

            # Check that master markdown has both slices
            master_content = Path(updated_state.master_md).read_text(encoding="utf-8")
            self.assertIn("Indledende afsnit oversættelse", master_content)
            self.assertIn("DRY RUN", master_content)
            self.assertIn("slice: p3-4", master_content)

    def test_cli_project_dispatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src_file = tmp / "Mantiq.txt"
            src_file.write_text(
                "علم المنطق آلة قانونية.\f"
                "تعصم مراعاتها الذهن عن الخطأ في الفكر.\f"
                "وهذا هو الغرض منه.\n",
                encoding="utf-8",
            )
            proj_file = tmp / "Mantiq.project.json"

            from translator.cli import main

            # 1. Project init via CLI
            main([
                "project",
                "init",
                str(src_file),
                "-p",
                str(proj_file),
                "-d",
                "mantiq",
                "-l",
                "da",
            ])
            self.assertTrue(proj_file.is_file())

            # 2. Project status via CLI
            main(["project", "status", str(proj_file)])

            # 3. Project run dry-run via root --project flag
            main([
                "--project",
                str(proj_file),
                "--pages",
                "1-2",
                "--dry-run",
            ])

            state = ProjectState.load(proj_file)
            self.assertEqual(state.next_page, 3)

            # 4. Project run with --all and --vowels
            main([
                "project",
                "run",
                str(proj_file),
                "--all",
                "--vowels",
                "--dry-run",
            ])
            state = ProjectState.load(proj_file)
            self.assertFalse(state.strip_harakat)


if __name__ == "__main__":
    unittest.main()
