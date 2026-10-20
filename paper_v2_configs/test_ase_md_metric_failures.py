"""Exercise ASE RDF/VDOS failure accounting without trajectory calculations."""

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import md_success
from model_display_names import normalize_display_key


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rdf = load_script("ase_failure_test_rdf", HERE / "rdfs/get-rdf-and-results-by-system-type-same-simulation-length.py")
vdos = load_script("ase_failure_test_vdos", HERE / "vdos/get_normalized_VDOS.py")
ASE = "mlip-trajs-ase-accelerated"
ACCELERATED = "mlip-trajs-torchsim-accelerated"
EAGER = "mlip-trajs-torchsim-eager"
AG = "bulkAg_600K_Kapil"
CU = "bulkCu_1000K_Kapil"
CRYSTAL = "anthracene_293K_Sharma_S"


class AseMetricFailureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "data"
        self.reference = self.data / "ref-trajs"
        self.failures = {}
        registry = patch.object(md_success, "load_failed_md_runs", side_effect=lambda: self.failures)
        registry.start()
        self.addCleanup(registry.stop)
        for system in (AG, CU, CRYSTAL):
            directory = self.reference / system
            directory.mkdir(parents=True)
            (directory / "traj.extxyz").touch()

    def job(self, model, system, *, source=ASE, timing="complete", h5=True):
        directory = self.data / source / system
        directory.mkdir(parents=True, exist_ok=True)
        trajectory = directory / f"nvt_{model}.h5"
        if h5:
            trajectory.touch()
        if timing != "missing":
            marker = directory / f"md_timing_{model}.csv"
            marker.write_text(
                "calculator,system,n_steps\n"
                f"{model.removesuffix('-ase')},{system},3\n"
                if timing == "complete" else ""
            )
        return trajectory

    def run_metrics(self, module, *, source=ASE, models=None):
        """Mock numerical kernels while exercising discovery, masks and CSV means."""
        args = SimpleNamespace(
            results_dir=self.root / module.__name__, models=models, systems=None,
            excluded_system_types=["molecular crystals"], dry_run=False,
            numerical_mlip_velocities=False, overwrite=False, pad_factor=1,
            chunk_size=32, e_min=None, e_max=None,
        )
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            if module is rdf:
                stack.enter_context(patch.object(module, "MLIP_TRAJ_DIRS", {source: self.data / source}))
                stack.enter_context(patch.object(module, "REF_TRAJ_BASE_DIR", self.reference))
                stack.enter_context(patch.object(module, "load_reference_trajectory", return_value=SimpleNamespace(n_frames=3)))
                load = stack.enter_context(patch.object(module, "load_mlip_trajectory", side_effect=lambda path, frames: path))
                stack.enter_context(patch.object(module, "h5_frame_count", return_value=3))
                stack.enter_context(patch.object(module, "compute_rdf", return_value=(np.array([0., 1.]), np.array([1., 1.]))))
                score = stack.enter_context(patch.object(module, "rdf_error", return_value=20.))
                stack.enter_context(patch.object(module, "save_rdf_csv"))
                module.process_source(args, source)
                directory = args.results_dir / source
                means = pd.read_csv(directory / "rdf_similarity_scores_same_simulation_length.csv")
                means = means.set_index("Calculator")["Mean RDF Error [%]"]
                rows = pd.concat([
                    pd.read_csv(path).assign(model=path.stem.removeprefix("rdf_similarity_scores_same_simulation_length_"))
                    for path in directory.glob("rdf_similarity_scores_same_simulation_length_*.csv")
                ]).rename(columns={"System": "system", "RDF_Error": "error"})
            else:
                stack.enter_context(patch.object(module, "load_reference_positions", return_value=np.zeros((8, 1, 3))))
                stack.enter_context(patch.object(module, "load_reference_atomic_numbers", return_value=np.array([29])))
                stack.enter_context(patch.object(module, "load_mlip_atomic_numbers", return_value=np.array([29])))
                stack.enter_context(patch.object(module, "h5_frame_count", return_value=8))
                load = stack.enter_context(patch.object(module, "load_mlip_samples", return_value=(np.zeros((8, 1, 3)), False)))
                spectrum = (np.array([0., 1.]), np.array([1., 1.]))
                stack.enter_context(patch.object(module, "spectrum_for_samples", return_value=spectrum))
                stack.enter_context(patch.object(module, "save_spectrum"))
                score = stack.enter_context(patch.object(module, "vdos_similarity", return_value=(20., 80.)))
                module.process_source(
                    args, source, data_dir=self.data, reference_dir=self.reference,
                    metadata={system: {"timestep": 1.} for system in (AG, CU, CRYSTAL)},
                )
                directory = args.results_dir / source
                means = pd.read_csv(directory / module.MODEL_OUTPUT).set_index("model")["vdos_error_percent"]
                rows = pd.read_csv(directory / module.PAIR_OUTPUT).rename(columns={"vdos_error_percent": "error"})
        return means, rows, load, score

    def test_native_ase_failed_jobs_retain_penalties_and_completed_jobs_are_scored(self):
        good = "orb-v2-force-only-ase"
        self.job(good, AG)
        self.job(good, CU, timing="empty")
        self.job(good, CRYSTAL)
        failed_models = {
            "mace-mp-0-compile-ase": {"timing": "empty"},
            "mace-mpa-0-compile-ase": {"h5": False},
            "pet-oam-xl-ase": {"timing": "missing"},
            "pet-omat-xl-ase": {},
        }
        for model, options in failed_models.items():
            self.job(model, AG, **options)
        self.failures[(ASE, AG, "pet-omat-xl-ase")] = "audited unphysical trajectory"

        for module in (rdf, vdos):
            with self.subTest(metric=module.__name__):
                means, rows, load, score = self.run_metrics(module)
                good_key = normalize_display_key(good)
                failed_keys = {normalize_display_key(model) for model in failed_models}
                self.assertEqual(means.loc[good_key], 60.)
                self.assertEqual(set(means.index), {good_key, *failed_keys})
                self.assertTrue(means.loc[list(failed_keys)].eq(100.).all())
                self.assertEqual(set(rows.system), {AG, CU})
                self.assertEqual(len(rows), 2 * (len(failed_models) + 1))
                self.assertEqual(rows.loc[(rows.model == good_key) & (rows.system == CU), "error"].tolist(), [100.])
                self.assertEqual(load.call_count, 1)
                self.assertEqual(load.call_args.args[0], self.data / ASE / AG / f"nvt_{good}.h5")
                self.assertEqual(score.call_count, 1)
                if module is vdos:
                    row = rows.loc[(rows.model == "pet-omat-xl") & (rows.system == AG)].iloc[0]
                    self.assertEqual(row.failure_reason, "audited unphysical trajectory")

    def test_completed_ase_model_is_not_averaged_with_a_stale_runner_alias(self):
        model = "orb-v3-direct-omat-force-only"
        for system in (AG, CU):
            self.job(model, system)
            self.job(f"{model}-ase", system, h5=False)
        for module in (rdf, vdos):
            with self.subTest(metric=module.__name__):
                means, rows, load, score = self.run_metrics(module, models=[f"{model}-ase"])
                self.assertEqual(means.to_dict(), {"orb-v3-direct-omat": 20.})
                self.assertEqual(len(rows), 2)
                self.assertTrue(rows.error.eq(20.).all())
                self.assertEqual(load.call_count, 2)
                self.assertTrue(all(call.args[0].name == f"nvt_{model}.h5" for call in load.call_args_list))
                self.assertEqual(score.call_count, 2)

    def test_explicit_missing_model_keeps_expected_system_coverage(self):
        model = "uma-s-omat-compile-force-only-ase"
        for module in (rdf, vdos):
            with self.subTest(metric=module.__name__):
                means, rows, load, score = self.run_metrics(module, models=[model])
                self.assertEqual(means.to_dict(), {model: 100.})
                self.assertEqual(set(rows.system), {AG, CU})
                load.assert_not_called()
                score.assert_not_called()

    def test_accelerated_torchsim_cohort_still_uses_borrowed_eager_runs(self):
        model = "eSEN-30M-OAM-force-only"
        self.job(model, AG, source=EAGER)
        self.job(model, CU, source=EAGER, timing="empty")
        for system in (AG, CU):
            self.job(model, system, source=ACCELERATED)
        for module in (rdf, vdos):
            with self.subTest(metric=module.__name__):
                means, _, load, _ = self.run_metrics(module, source=ACCELERATED)
                self.assertEqual(means.to_dict(), {model: 60.})
                self.assertEqual(load.call_count, 1)
                self.assertEqual(load.call_args.args[0], self.data / EAGER / AG / f"nvt_{model}.h5")


if __name__ == "__main__":
    unittest.main()
