#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9", "pandas>=2.2"]
# ///
"""Plot mean matched-length VDOS error for one trajectory source."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


PAPER_V2_CONFIG_DIR = Path(__file__).resolve().parents[1]
if str(PAPER_V2_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_V2_CONFIG_DIR))
from model_display_names import display_model_name


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "results" / "vdos_model_mean_ev_normalized_same_simulation_length.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "plots" / "plot_vdos_model_errors.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-means-file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--title")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.model_means_file)
    required = {"model", "vdos_error_percent"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {args.model_means_file}: {sorted(missing)}")
    frame["vdos_error_percent"] = pd.to_numeric(frame["vdos_error_percent"], errors="coerce")
    frame = frame.dropna(subset=["vdos_error_percent"]).sort_values("vdos_error_percent")
    if frame.empty:
        raise ValueError(f"No finite VDOS errors in {args.model_means_file}")
    frame["model_display"] = frame["model"].map(display_model_name)

    height = max(3.5, 0.28 * len(frame) + 1.2)
    fig, ax = plt.subplots(figsize=(7.0, height))
    ax.barh(frame["model_display"], frame["vdos_error_percent"], color="#4c78a8")
    ax.invert_yaxis()
    ax.set_xlabel("Mean VDOS error [%]")
    ax.set_ylabel("Model")
    ax.set_title(args.title or "Matched-length VDOS error")
    ax.grid(axis="x", alpha=0.35, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved VDOS model-error plot to {args.output_file}")


if __name__ == "__main__":
    main()
