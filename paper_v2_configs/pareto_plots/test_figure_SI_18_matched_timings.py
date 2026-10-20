"""Check SI18 pairs timing workloads without changing accuracy penalties."""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import figure_7
import figure_SI_17 as timing


COPPER = "bulkCu_1000K_Kapil"
GOLD = "bulkAu_1500K_Kapil"
HYDROGEN = "hydrogen_1050K_Cheng"
CRYSTAL = "naphthalene_295K_Sharma_S"


class FigureSI18MatchedTimingsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.timings_dirs = {}
        for source in (timing.TORCHSIM_SOURCE, timing.ASE_SOURCE):
            directory = self.root / source
            directory.mkdir()
            self.timings_dirs[source] = directory
        self.failures = {}
        for target in ("figure_SI_17.load_failed_md_runs", "md_success.load_failed_md_runs"):
            patcher = patch(target, return_value=self.failures)
            patcher.start()
            self.addCleanup(patcher.stop)

    def add_run(self, source, model, system=COPPER, seconds=0.001, steps=100,
                **fields):
        folder = self.timings_dirs[source] / system
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"md_timing_{model}.csv"
        row = {
            "calculator": model.removesuffix("-ase"),
            "system": system,
            "n_steps": steps,
            "elapsed_seconds": seconds * steps,
            "seconds_per_step": seconds,
            "time_step_fs": 1,
            "record_interval": 1,
            "engine": source,
        }
        row.update(fields)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        path.with_name(f"nvt_{model}.h5").touch()
        return path

    @staticmethod
    def metric_frame(models=("orb-v2", "orb-v2")):
        return pd.DataFrame({
            "model": models,
            "dataset": [timing.TORCHSIM_SOURCE, timing.ASE_SOURCE],
            "mean_time_per_step_ms": [80.0, 90.0],
            "RDF Error [%]": [10.0, 100.0],
            "VDOS Error [%]": [20.0, 100.0],
            "Pressure Error [%]": [30.0, 100.0],
            "Combined Error [%]": [20.0, 100.0],
        })

    def apply(self, frame, include_molecular_crystals=False):
        return figure_7.apply_matched_runner_timings(
            frame,
            timings_dirs=self.timings_dirs,
            include_molecular_crystals=include_molecular_crystals,
        )

    def test_pet_aliases_drop_unmatched_hydrogen_but_keep_accuracy_penalties(self):
        torchsim_model = "pet-oam-xl-force-only-torchscript"
        ase_model = "pet-oam-xl-ase"
        for system, steps, ts_seconds, ase_seconds in (
            (COPPER, 100, 0.002, 0.003),
            (GOLD, 1000, 0.006, 0.005),
        ):
            self.add_run(timing.TORCHSIM_SOURCE, torchsim_model, system,
                         ts_seconds, steps, engine="torch-sim+pet-torchscript-force-only")
            self.add_run(timing.ASE_SOURCE, ase_model, system, ase_seconds,
                         steps, engine="ase+upet-0.2.6-torchscript-force-only")
        self.add_run(timing.TORCHSIM_SOURCE, torchsim_model, HYDROGEN,
                     seconds=0.970, engine="torch-sim+pet-torchscript-force-only")
        frame = self.metric_frame(("pet-oam-xl-torchscript", "pet-oam-xl"))
        original = frame.copy(deep=True)

        result = self.apply(frame)

        self.assertEqual(result["mean_time_per_step_ms"].tolist(), [4.0, 4.0])
        self.assertEqual(result["n_timing_systems"].tolist(), [2, 2])
        self.assertEqual(result["timing_systems"].tolist(),
                         [";".join(sorted((COPPER, GOLD)))] * 2)
        self.assertEqual(result["timing_basis"].tolist(), ["matched systems"] * 2)
        # SI18's accuracy axis must keep the benchmark's failed-MD penalties.
        for column in original.columns.difference(["mean_time_per_step_ms"]):
            pd.testing.assert_series_equal(result[column], original[column])
        pd.testing.assert_frame_equal(frame, original)

    def test_models_present_in_only_one_plotted_cohort_keep_original_timings(self):
        frame = self.metric_frame(("esen-30m-oam", "orb-v2"))
        # Native files in the other runner do not make a model part of SI18
        # unless its complete metric row is actually plotted there.
        for source in (timing.TORCHSIM_SOURCE, timing.ASE_SOURCE):
            self.add_run(source, "orb-v2-force-only", seconds=0.002)
        result = self.apply(frame)

        pd.testing.assert_frame_equal(result[frame.columns], frame)
        self.assertTrue(result["n_timing_systems"].isna().all())
        self.assertEqual(result["timing_basis"].tolist(), ["runner-specific"] * 2)

    def test_different_md_workloads_are_not_averaged_together(self):
        for source, seconds in ((timing.TORCHSIM_SOURCE, 0.002),
                                (timing.ASE_SOURCE, 0.004)):
            self.add_run(source, "orb-v2-force-only", seconds=seconds)
            for system in ("different_steps", "different_timestep", "different_interval"):
                fields = {}
                steps = 100
                if source == timing.ASE_SOURCE:
                    if system == "different_steps":
                        steps = 50
                    elif system == "different_timestep":
                        fields["time_step_fs"] = 2
                    else:
                        fields["record_interval"] = 2
                self.add_run(source, "orb-v2-force-only", system=system,
                             seconds=1.0, steps=steps, **fields)
        result = self.apply(self.metric_frame())

        self.assertEqual(result["mean_time_per_step_ms"].tolist(), [2.0, 4.0])
        self.assertEqual(result["n_timing_systems"].tolist(), [1, 1])
        self.assertEqual(result["timing_systems"].tolist(), [COPPER, COPPER])

    def test_shared_models_without_matched_workloads_are_removed_from_both(self):
        self.add_run(timing.TORCHSIM_SOURCE, "orb-v2-force-only", system=COPPER)
        self.add_run(timing.ASE_SOURCE, "orb-v2-force-only-ase", system=GOLD)
        frame = pd.concat([
            self.metric_frame(),
            self.metric_frame(("esen-30m-oam", "mattersim-v1-5m-compile")),
        ], ignore_index=True)

        result = self.apply(frame)

        self.assertEqual(result["model"].tolist(),
                         ["esen-30m-oam", "mattersim-v1-5m-compile"])
        self.assertEqual(result["mean_time_per_step_ms"].tolist(), [80.0, 90.0])
        self.assertEqual(result["timing_basis"].tolist(), ["runner-specific"] * 2)

    def test_failed_and_molecular_crystal_runs_do_not_enter_default_mean(self):
        for source, seconds in ((timing.TORCHSIM_SOURCE, 0.002),
                                (timing.ASE_SOURCE, 0.004)):
            for system in (COPPER, CRYSTAL, "unstable"):
                self.add_run(source, "orb-v2-force-only", system=system,
                             seconds=seconds if system == COPPER else 1.0)
        self.failures[(timing.ASE_SOURCE, "unstable", "orb-v2-force-only")] = "Unphysical run"

        result = self.apply(self.metric_frame())

        self.assertEqual(result["mean_time_per_step_ms"].tolist(), [2.0, 4.0])
        self.assertEqual(result["n_timing_systems"].tolist(), [1, 1])
        with_crystals = self.apply(self.metric_frame(), include_molecular_crystals=True)
        self.assertEqual(with_crystals["mean_time_per_step_ms"].tolist(), [501.0, 502.0])
        self.assertEqual(with_crystals["n_timing_systems"].tolist(), [2, 2])
        self.assertEqual(with_crystals["timing_systems"].tolist(),
                         [";".join(sorted((COPPER, CRYSTAL)))] * 2)


if __name__ == "__main__":
    unittest.main()
