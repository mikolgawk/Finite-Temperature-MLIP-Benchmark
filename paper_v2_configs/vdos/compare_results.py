#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=1.26", "pandas>=2.2"]
# ///
"""Compare the corrected VDOS results with the original analysis, without plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCES = ("mlip-trajs-torchsim-eager", "mlip-trajs-torchsim-accelerated")
PAIRS = "vdos_pair_errors_ev_normalized_same_simulation_length.csv"
MODELS = "vdos_model_mean_ev_normalized_same_simulation_length.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", choices=SOURCES, dest="sources")
    parser.add_argument("--original-results", type=Path, default=HERE.parent / "vdos_legacy/results")
    parser.add_argument("--updated-results", type=Path, default=HERE / "results")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results/comparison")
    args = parser.parse_args()
    sources = args.sources or list(SOURCES)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs, models, summary = [], [], {}
    for source in sources:
        old_models = pd.read_csv(args.original_results / source / MODELS)
        new_models = pd.read_csv(args.updated_results / source / MODELS)
        if "trajectory_source" not in old_models:
            if source != "mlip-trajs-torchsim-eager":
                raise ValueError(f"Missing original hybrid-cohort provenance for {source}")
            old_models["trajectory_source"] = source
        old_pairs = pd.read_csv(args.original_results / source / PAIRS)
        new_pairs = pd.read_csv(args.updated_results / source / PAIRS)
        if "trajectory_source" not in old_pairs:
            old_pairs = old_pairs.merge(
                old_models[["source", "model", "trajectory_source"]],
                on=["source", "model"], how="left", validate="many_to_one",
            )
        paired = old_pairs.merge(
            new_pairs, on=["source", "model", "system"], how="outer",
            suffixes=("_original", "_updated"), indicator=True, validate="one_to_one",
        )
        if not paired["_merge"].eq("both").all():
            raise ValueError(f"Original and updated model/system coverage differ for {source}")
        if not paired.trajectory_source_original.eq(paired.trajectory_source_updated).all():
            raise ValueError(f"Physical trajectory sources changed for {source}")
        old_failed = paired.failure_reason_original.fillna("").ne("")
        new_failed = paired.failure_reason_updated.fillna("").ne("")
        if not old_failed.equals(new_failed):
            raise ValueError(f"Failure classification changed for {source}; inspect pair tables")
        if not paired.loc[new_failed, "vdos_error_percent_updated"].eq(100.0).all():
            raise ValueError(f"Missing 100% failure penalties for {source}")
        if not np.isfinite(paired.vdos_error_percent_updated).all():
            raise ValueError(f"Nonfinite updated errors for {source}")
        paired["error_change_percentage_points"] = (
            paired.vdos_error_percent_updated - paired.vdos_error_percent_original
        )
        pairs.append(paired.drop(columns="_merge"))

        merged = old_models.merge(
            new_models, on=["source", "model"], suffixes=("_original", "_updated"),
            how="outer", indicator=True, validate="one_to_one",
        )
        if not merged["_merge"].eq("both").all():
            raise ValueError(f"Model summary coverage changed for {source}")
        if not merged.n_systems_original.eq(merged.n_systems_updated).all():
            raise ValueError(f"Model averaging denominators changed for {source}")
        merged["error_change_percentage_points"] = (
            merged.vdos_error_percent_updated - merged.vdos_error_percent_original
        )
        models.append(merged.drop(columns="_merge"))
        summary[source] = {
            "models": len(new_models),
            "systems": int(new_pairs.system.nunique()),
            "model_system_pairs": len(new_pairs),
            "failure_penalties": int(new_failed.sum()),
            "coverage_and_failures_unchanged": True,
            "original_median_model_error_percent": float(old_models.vdos_error_percent.median()),
            "updated_median_model_error_percent": float(new_models.vdos_error_percent.median()),
            "mean_model_error_change_percentage_points": float(merged.error_change_percentage_points.mean()),
        }
    pd.concat(pairs, ignore_index=True).to_csv(args.output_dir / "per_system_comparison.csv", index=False)
    pd.concat(models, ignore_index=True).to_csv(args.output_dir / "model_comparison.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
