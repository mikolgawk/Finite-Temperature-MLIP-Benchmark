#!/usr/bin/env python3
"""Collect the corrected plots into per-cohort PDFs and a downloadable archive.

Combining PDF collections requires Poppler's ``pdfunite``. ``--archive-only``
refreshes the archive from existing collections without invoking PDF tools.
Native ASE summaries and the standalone SI18 comparison are included when
available. All individual figures remain in ``plots/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile


HERE = Path(__file__).resolve().parent
SOURCES = ("mlip-trajs-torchsim-eager", "mlip-trajs-torchsim-accelerated")
ASE_SOURCE = "mlip-trajs-ase-accelerated"
COMPARISON_DIR = "torchsim_vs_ase_accelerated"
PAGES = (
    ("figure_5.pdf", "VDOS overview and representative spectra"),
    ("figure_6.pdf", "Correlations with outlier labels"),
    ("figure_6_all_labels.pdf", "Correlations with all model labels"),
    ("figure_SI_7_8_9_rdf.pdf", "RDF correlations"),
    ("figure_SI_7_8_9_pressure.pdf", "Pressure correlations"),
    ("figure_SI_7_8_9_vdos.pdf", "VDOS correlations"),
    ("figure_SI_16.pdf", "VDOS accuracy versus MD cost"),
    ("figure_7.pdf", "Combined accuracy versus MD cost"),
    ("figure_SI_13.pdf", "Combined Lehmer-mean accuracy versus MD cost"),
    ("figure_SI_11.pdf", "Pressure versus VDOS error"),
    ("figure_SI_12.pdf", "RDF versus VDOS error"),
    ("plot_vdos_results.pdf", "Model VDOS error summary"),
)


def comparison_figure(plots_dir: Path, results_dir: Path) -> list[tuple[Path, str]]:
    """Include SI18 only as a complete figure/metrics/frontier set."""
    figure = plots_dir / COMPARISON_DIR / "figure_SI_18.pdf"
    results = results_dir / COMPARISON_DIR
    required = [figure, *(results / name for name in (
        "figure_SI_18_metrics.csv", "figure_SI_18_frontier.csv",
    ))]
    comparison_results = [path for path in results.glob("figure_SI_18*")
                          if path.is_file() and path.suffix in {".csv", ".json"}]
    if not figure.is_file() and not comparison_results:
        return []
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Incomplete SI18 comparison; finish its figure and CSV exports before packaging. "
            "Missing: " + ", ".join(map(str, missing))
        )
    return [(figure, f"plots/{COMPARISON_DIR}/{figure.name}")]


def build_archive(
    plots_dir: Path, results_dir: Path, output_dir: Path, sources: list[str],
) -> Path:
    """Package existing artifacts, preserving the previous ZIP if writing fails."""
    figures = [(plots_dir / source / name, f"plots/{source}/{name}")
               for source in sources for name, _ in PAGES]
    figures.extend(comparison_figure(plots_dir, results_dir))
    books = [output_dir / f"vdos_updated_{source.removeprefix('mlip-trajs-torchsim-')}.pdf"
             for source in sources]
    index = output_dir / "plot_index.md"
    entries = [*figures, *((path, path.name) for path in (*books, index))]
    for path, _ in entries:
        if not path.is_file():
            raise FileNotFoundError(path)
    for name in ("README.md", "analysis_summary.md", "methods_vdos.tex"):
        path = HERE / name
        if path.is_file():
            entries.append((path, name))
    diagnostic_directories = [
        path.name for path in results_dir.iterdir()
        if path.is_dir() and (path.name.startswith("comparison") or path.name == "sensitivity")
    ]
    result_groups = dict.fromkeys((*sources, ASE_SOURCE, COMPARISON_DIR,
                                 *sorted(diagnostic_directories)))
    for group in result_groups:
        for path in sorted((results_dir / group).glob("*")):
            if path.is_file() and path.suffix in {".csv", ".json"}:
                entries.append((path, f"results/{group}/{path.name}"))
    archive = output_dir / "vdos_updated_plots_and_metrics.zip"
    with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".zip", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as package:
            for path, target in entries:
                package.write(path, target)
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", choices=SOURCES, dest="sources")
    parser.add_argument("--plots-dir", type=Path, default=HERE / "plots")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    parser.add_argument("--output-dir", type=Path, default=HERE / "deliverables")
    parser.add_argument("--archive-only", action="store_true",
                        help="refresh the ZIP using existing PDF collections without invoking pdfunite")
    args = parser.parse_args()
    pdfunite = None if args.archive_only else shutil.which("pdfunite")
    if not args.archive_only and pdfunite is None:
        parser.error("pdfunite is required to combine PDFs; install Poppler or use the individual plots")
    sources = list(dict.fromkeys(args.sources or SOURCES))
    comparison = comparison_figure(args.plots_dir, args.results_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        selected = [args.plots_dir / source / name for name, _ in PAGES]
        for path in selected:
            if not path.is_file():
                raise FileNotFoundError(path)
        cohort = source.removeprefix("mlip-trajs-torchsim-")
        book = args.output_dir / f"vdos_updated_{cohort}.pdf"
        if args.archive_only:
            if not book.is_file():
                raise FileNotFoundError(f"Existing PDF collection required by --archive-only: {book}")
            continue
        with tempfile.NamedTemporaryFile(dir=args.output_dir, suffix=".pdf", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            subprocess.run([pdfunite, *(str(path) for path in selected), str(temporary)], check=True)
            temporary.replace(book)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Saved {book}")
    index = args.output_dir / "plot_index.md"
    index.write_text(
        "# Corrected VDOS plot collections\n\n"
        "The eager and accelerated PDF collections use this page order.\n\n"
        "| Page | Figure | Contents |\n|---:|---|---|\n"
        + "".join(f"| {page} | {name} | {label} |\n"
                  for page, (name, label) in enumerate(PAGES, 1))
        + "\nIndividual vector PDFs and result summaries are included in the archive. "
        "Raw trajectories and spectrum caches remain in the analysis workspace.\n"
        + ("\nThe standalone Figure SI18 compares accelerated ASE and TorchSim. "
           "Its PDF and comparison tables are included alongside the collections.\n"
           if comparison else "")
    )
    archive = build_archive(args.plots_dir, args.results_dir, args.output_dir, sources)
    print(f"Saved {archive}")


if __name__ == "__main__":
    main()
