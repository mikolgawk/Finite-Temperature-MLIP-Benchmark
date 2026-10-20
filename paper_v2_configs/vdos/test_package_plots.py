"""Check deliverable contents without combining or generating PDF figures."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import package_plots as packaging


class PackagePlotsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.plots = self.root / "plots"
        self.results = self.root / "results"
        self.output = self.root / "deliverables"
        self.results.mkdir()
        self.output.mkdir()
        for source in packaging.SOURCES:
            for name, _ in packaging.PAGES:
                self.write(self.plots / source / name, f"individual {source} {name}")
            self.write(self.results / source / "model_means.csv", "model,error\norb-v2,20\n")
        for group in ("comparison", "comparison_periodogram", "sensitivity"):
            self.write(self.results / group / "summary.json", '{"complete":true}')
            self.write(self.results / group / "systems.csv", "system,error\nAg,10\n")
            self.write(self.results / group / "private.txt", "not a summary")
        self.docs = self.root / "analysis"
        for name in ("README.md", "analysis_summary.md", "methods_vdos.tex"):
            self.write(self.docs / name, name)

    @staticmethod
    def write(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def optional_inputs(self):
        self.write(self.results / packaging.ASE_SOURCE / "vdos_model_mean_ev_normalized_same_simulation_length.csv",
                   "model,vdos_error_percent\npet-oam-xl,18\n")
        self.write(self.results / packaging.ASE_SOURCE / "metadata.json", '{"method":"refined"}')
        self.write(self.results / packaging.ASE_SOURCE / "vdos_same_simulation_length_saved/mlip/pet-oam-xl/Ag.csv",
                   "frequency,spectrum\n1,1\n")
        self.write(self.plots / packaging.COMPARISON_DIR / "figure_SI_18.pdf", "existing SI18 PDF")
        for name in ("figure_SI_18_metrics.csv", "figure_SI_18_frontier.csv"):
            self.write(self.results / packaging.COMPARISON_DIR / name, "model,error\norb-v2,20\n")
        self.write(self.results / packaging.COMPARISON_DIR / "figure_SI_18_metadata.json",
                   '{"separate_frontiers":true}')

    def run_packaging(self, *extra):
        argv = ["package_plots.py", "--plots-dir", str(self.plots),
                "--results-dir", str(self.results), "--output-dir", str(self.output), *extra]

        def combine(command, **kwargs):
            self.assertEqual(command[0], "/test/pdfunite")
            self.assertTrue(kwargs.get("check"))
            self.assertEqual(len(command), len(packaging.PAGES) + 2)
            Path(command[-1]).write_text("existing pages combined by mocked pdfunite")

        with (patch.object(sys, "argv", argv),
              patch.object(packaging, "HERE", self.docs),
              patch.object(packaging.shutil, "which", return_value="/test/pdfunite") as which,
              patch.object(packaging.subprocess, "run", side_effect=combine) as unite,
              contextlib.redirect_stdout(io.StringIO())):
            packaging.main()
        return which, unite

    @property
    def archive(self):
        return self.output / "vdos_updated_plots_and_metrics.zip"

    def test_archive_keeps_both_collections_and_adds_native_ase_and_complete_si18(self):
        self.optional_inputs()
        _, unite = self.run_packaging()
        self.assertEqual(unite.call_count, 2)
        with ZipFile(self.archive) as archive:
            names = archive.namelist()
            self.assertEqual(len(names), len(set(names)))
            for source in packaging.SOURCES:
                cohort = source.removeprefix("mlip-trajs-torchsim-")
                self.assertIn(f"vdos_updated_{cohort}.pdf", names)
                self.assertIn(f"results/{source}/model_means.csv", names)
                for name, _ in packaging.PAGES:
                    self.assertIn(f"plots/{source}/{name}", names)
            self.assertIn(f"results/{packaging.ASE_SOURCE}/metadata.json", names)
            self.assertIn(f"results/{packaging.ASE_SOURCE}/vdos_model_mean_ev_normalized_same_simulation_length.csv", names)
            self.assertEqual(archive.read(f"plots/{packaging.COMPARISON_DIR}/figure_SI_18.pdf"), b"existing SI18 PDF")
            for name in ("figure_SI_18_metrics.csv", "figure_SI_18_frontier.csv", "figure_SI_18_metadata.json"):
                self.assertIn(f"results/{packaging.COMPARISON_DIR}/{name}", names)
            for group in ("comparison", "comparison_periodogram", "sensitivity"):
                self.assertIn(f"results/{group}/summary.json", names)
                self.assertIn(f"results/{group}/systems.csv", names)
            self.assertFalse(any("spectrum" in name or "_saved/" in name or name.endswith("private.txt") for name in names))
            for name in ("plot_index.md", "README.md", "analysis_summary.md", "methods_vdos.tex"):
                self.assertIn(name, names)
            self.assertIn(b"standalone Figure SI18", archive.read("plot_index.md"))

    def test_ase_and_si18_are_optional_when_they_have_not_been_produced(self):
        self.run_packaging()
        with ZipFile(self.archive) as archive:
            names = archive.namelist()
            self.assertFalse(any(packaging.ASE_SOURCE in name or packaging.COMPARISON_DIR in name for name in names))
            self.assertIn("vdos_updated_eager.pdf", names)
            self.assertIn("vdos_updated_accelerated.pdf", names)
            self.assertIn("results/sensitivity/summary.json", names)

    def test_partial_si18_fails_before_replacing_existing_bundle_or_pdf_books(self):
        for missing_name in ("figure_SI_18.pdf", "figure_SI_18_metrics.csv", "figure_SI_18_frontier.csv"):
            with self.subTest(missing=missing_name):
                self.optional_inputs()
                directory = (self.plots if missing_name.endswith(".pdf") else self.results) / packaging.COMPARISON_DIR
                (directory / missing_name).unlink()
                self.archive.write_bytes(b"previous complete archive")
                book = self.output / "vdos_updated_eager.pdf"
                book.write_bytes(b"previous eager book")
                with (patch.object(packaging.subprocess, "run", side_effect=AssertionError("Must validate before combining")),
                      self.assertRaisesRegex(FileNotFoundError, "Incomplete SI18 comparison.*" + missing_name)):
                    self.run_packaging()
                self.assertEqual(self.archive.read_bytes(), b"previous complete archive")
                self.assertEqual(book.read_bytes(), b"previous eager book")

    def test_archive_only_refresh_never_invokes_pdf_tools_or_rewrites_books(self):
        self.optional_inputs()
        before = {}
        for cohort in ("eager", "accelerated"):
            book = self.output / f"vdos_updated_{cohort}.pdf"
            book.write_bytes(f"original {cohort} collection".encode())
            before[book] = (book.read_bytes(), book.stat().st_mtime_ns)
        which, unite = self.run_packaging("--archive-only")
        which.assert_not_called()
        unite.assert_not_called()
        self.assertEqual(before, {book: (book.read_bytes(), book.stat().st_mtime_ns) for book in before})
        with ZipFile(self.archive) as archive:
            self.assertEqual(archive.read("vdos_updated_eager.pdf"), before[self.output / "vdos_updated_eager.pdf"][0])
            self.assertIn(f"plots/{packaging.COMPARISON_DIR}/figure_SI_18.pdf", archive.namelist())

    def test_archive_only_requires_an_existing_selected_collection(self):
        self.archive.write_bytes(b"previous complete archive")
        with self.assertRaisesRegex(FileNotFoundError, "Existing PDF collection required"):
            self.run_packaging("--archive-only", "--source", packaging.SOURCES[0])
        self.assertEqual(self.archive.read_bytes(), b"previous complete archive")
        self.write(self.output / "vdos_updated_eager.pdf", "eager only")
        self.run_packaging("--archive-only", "--source", packaging.SOURCES[0])
        with ZipFile(self.archive) as archive:
            self.assertIn("vdos_updated_eager.pdf", archive.namelist())
            self.assertNotIn("vdos_updated_accelerated.pdf", archive.namelist())

    def test_failed_archive_write_preserves_previous_zip(self):
        self.run_packaging()
        before = self.archive.read_bytes()
        with (patch.object(packaging.ZipFile, "write", side_effect=OSError("interrupted read")),
              self.assertRaisesRegex(OSError, "interrupted read")):
            packaging.build_archive(self.plots, self.results, self.output, list(packaging.SOURCES))
        self.assertEqual(self.archive.read_bytes(), before)
        self.assertEqual(list(self.output.glob("*.zip")), [self.archive])


if __name__ == "__main__":
    unittest.main()
