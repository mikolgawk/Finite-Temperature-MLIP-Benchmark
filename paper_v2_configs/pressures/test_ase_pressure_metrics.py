"""Exercise native ASE pressure aggregation without running MD or plotting."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import get_model_pressure_errors as pressure


SOURCE = "mlip-trajs-ase-accelerated"
AG = "bulkAg_600K_Kapil"
CU = "bulkCu_1000K_Kapil"
HYDROGEN = "H_1050K_Rupp_QE"
ALLOY = "bulkPt3Co_300K_J.Kioseoglou_VASP"
INTERFACE = "Pt111w24H2O_380K_Heenen_VASP"
CRYSTAL = "naphthalene_295K_Sharma_S"
NO_STRESS = "bulkCuAu_500K-Artrith_VASP"


class AsePressureMetricTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "data"
        self.reference = self.data / "ref-trajs"
        self.reference.mkdir(parents=True)
        self.metadata = {}
        self.inputs = self.root / "evaluations" / SOURCE
        (self.inputs / "references").mkdir(parents=True)
        self.output = self.root / "results" / SOURCE
        self.failures = {}
        for target in ("md_success.load_failed_md_runs", "get_model_pressure_errors.load_failed_md_runs"):
            registry = patch(target, return_value=self.failures)
            registry.start()
            self.addCleanup(registry.stop)

    def reference_system(self, system, stress_stride=1):
        directory = self.reference / system
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "traj.extxyz").touch()
        self.metadata[system] = {"timestep": 1.0, "stress_print_stride": stress_stride}
        (self.reference / "md_metadata.json").write_text(json.dumps(self.metadata))

    def md_run(self, model, system, *, marker="complete", h5=True):
        if system not in self.metadata:
            self.reference_system(system)
        directory = self.data / SOURCE / system
        directory.mkdir(parents=True, exist_ok=True)
        trajectory = directory / f"nvt_{model}.h5"
        if h5:
            trajectory.touch()
        if marker != "missing":
            marker_path = directory / f"md_timing_{model}.csv"
            marker_path.write_text(
                "calculator,system,n_steps\n"
                f"{model.removesuffix('-ase')},{system},100\n"
                if marker == "complete" else "calculator,system,n_steps\n"
            )
        return trajectory

    def evaluation(self, model, errors, *, pressure_offset=0.0):
        rows = []
        for system in errors:
            if system not in self.metadata:
                self.reference_system(system)
            for index, value in enumerate(np.arange(16) / 10):
                rows.append({"trajectory_file": f"/other-machine/{system}/nvt_{model}.h5",
                             "frame_index": index, "pressure_GPa": value})
        reference = pd.DataFrame(rows)
        prediction = reference.assign(pressure_GPa=reference.pressure_GPa + pressure_offset)
        path = self.inputs / f"{model}{pressure.DEFAULT_MODEL_FILE_SUFFIX}"
        prediction.to_csv(path, index=False)
        reference.to_csv(self.inputs / "references" / f"{model}.csv", index=False)
        pd.DataFrame({
            "trajectory_file": [f"/other-machine/{system}/nvt_{model}.h5" for system in errors],
            "absolute_mean_error_GPa": list(errors.values()),
        }).to_csv(self.inputs / f"{model}{pressure.TRAJECTORY_SUMMARY_SUFFIX}", index=False)
        return path

    def compute(self, *, cached=None):
        with contextlib.redirect_stdout(io.StringIO()):
            result = pressure.compute_pressure_metric(
                self.inputs, None, pressure.DEFAULT_MODEL_FILE_SUFFIX, 8,
                self.output / "pairs.csv", self.output / "systems.csv",
                self.output / "models.csv", self.output / "types.csv", cached,
                source=SOURCE, md_data_dir=self.data,
            )
        return result.set_index("model"), pd.read_csv(self.output / "pairs.csv")

    def test_failed_native_md_gets_histogram_penalties_and_is_excluded_from_mae(self):
        model = "orb-v2-force-only-ase"
        self.md_run(model, AG)
        self.md_run(model, CU, marker="missing")
        self.md_run(model, HYDROGEN, marker="empty")
        self.md_run(model, ALLOY)
        self.failures[(SOURCE, ALLOY, model)] = "Audited unphysical trajectory"
        self.evaluation("orb-v2", {AG: 2.0, CU: 1000.0, HYDROGEN: 2000.0, ALLOY: 3000.0})
        cached = self.root / "old-mae.csv"
        cached.write_text("model,error_GPa\norb-v2,99999\n")
        original_cache = cached.read_bytes()

        models, pairs = self.compute(cached=cached)
        self.assertEqual(set(pairs.system), {AG, CU, HYDROGEN, ALLOY})
        self.assertEqual(len(pairs), 4)
        indexed = pairs.set_index("system")
        self.assertEqual(indexed.loc[AG, "pressure_error_percent"], 0.0)
        self.assertTrue(indexed.loc[[CU, HYDROGEN, ALLOY], "pressure_error_percent"].eq(100.0).all())
        self.assertTrue(indexed.loc[[CU, HYDROGEN, ALLOY], "n_mlip_frames"].eq(0).all())
        self.assertEqual(indexed.loc[ALLOY, "failure_reason"], "Audited unphysical trajectory")
        self.assertEqual(models.loc["orb-v2", "pressure_error_percent"], 75.0)
        self.assertEqual(models.loc["orb-v2", "pressure_mae_GPa"], 2.0)
        comparison = pd.read_csv(self.output / "model_mean_pressure_comparison.csv")
        self.assertEqual(comparison.error_GPa.tolist(), [2.0])
        self.assertEqual(cached.read_bytes(), original_cache)
        self.assertEqual(set(pairs.backend), {"ase"})
        self.assertEqual(set(pairs["mode"]), {"md_accelerated"})
        self.assertEqual(set(pairs.trajectory_source), {SOURCE})

    def test_missing_h5_overrides_saved_pressure_results_and_dimensional_errors(self):
        self.md_run("orb-v2-force-only-ase", AG)
        self.md_run("orb-v2-force-only-ase", CU, h5=False)
        self.evaluation("orb-v2", {AG: 3.0, CU: 1e8})
        models, pairs = self.compute()
        self.assertEqual(pairs.set_index("system").loc[CU, "pressure_error_percent"], 100.0)
        self.assertEqual(models.loc["orb-v2", "pressure_error_percent"], 50.0)
        self.assertEqual(models.loc["orb-v2", "pressure_mae_GPa"], 3.0)

    def test_current_evaluator_files_replace_obsolete_ase_alias_without_double_counting(self):
        raw_model = "orb-v3-direct-omat-force-only"
        self.md_run(raw_model, AG)
        self.md_run(raw_model + "-ase", AG, h5=False, marker="empty")
        current = self.evaluation("orb-v3-direct-omat", {AG: 3.0})
        self.evaluation("orb-v3-direct-omat-ase", {AG: 9999.0}, pressure_offset=1e8)
        self.assertEqual(pressure.pressure_model_files(self.inputs, pressure.DEFAULT_MODEL_FILE_SUFFIX),
                         [current])
        models, pairs = self.compute()
        self.assertEqual(models.index.tolist(), ["orb-v3-direct-omat"])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs.model_file.tolist(), [str(current)])
        self.assertEqual(models.loc["orb-v3-direct-omat", "pressure_error_percent"], 0.0)
        self.assertEqual(models.loc["orb-v3-direct-omat", "pressure_mae_GPa"], 3.0)

    def test_successful_md_with_pending_stress_evaluation_is_not_a_pressure_model(self):
        self.md_run("orb-v2-force-only-ase", AG)
        self.md_run("pet-oam-xl-ase", AG)
        self.evaluation("orb-v2", {AG: 1.0})
        models, pairs = self.compute()
        self.assertEqual(models.index.tolist(), ["orb-v2"])
        self.assertEqual(pairs.mlip_model.tolist(), ["orb-v2"])
        self.assertEqual(models.loc["orb-v2", "pressure_error_percent"], 0.0)

    def test_explicit_pending_pet_request_cannot_create_fake_pressure_penalties(self):
        self.md_run("orb-v2-force-only-ase", AG)
        self.md_run("pet-oam-xl-ase", AG)
        self.evaluation("orb-v2", {AG: 1.0})
        requested = {"pet-oam-xl-ase"}
        pairs = pressure.build_pair_rows(
            self.inputs, None, pressure.DEFAULT_MODEL_FILE_SUFFIX, 8,
            models=requested, allow_empty=True,
        )
        selected = pressure.add_failed_system_penalties(
            pairs, self.inputs, SOURCE, requested, {"molecular crystals"}, 8,
            md_data_dir=self.data,
        )
        self.assertTrue(selected.empty)

    def test_interface_crystals_and_reference_without_stress_are_excluded(self):
        self.reference_system(NO_STRESS, stress_stride=None)
        for system in (AG, INTERFACE, CRYSTAL, NO_STRESS):
            self.md_run("orb-v2-force-only-ase", system)
        self.evaluation("orb-v2", {AG: 2.0, INTERFACE: 1000.0, CRYSTAL: 2000.0, NO_STRESS: 3000.0})
        models, pairs = self.compute()
        self.assertEqual(pairs.system.tolist(), [AG])
        self.assertEqual(models.loc["orb-v2", "pressure_error_percent"], 0.0)
        self.assertEqual(models.loc["orb-v2", "pressure_mae_GPa"], 2.0)


if __name__ == "__main__":
    unittest.main()
