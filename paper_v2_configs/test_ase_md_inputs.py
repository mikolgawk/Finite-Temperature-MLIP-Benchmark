"""Validate ASE runner aliases and completion markers without creating figures."""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from md_success import discover_md_trajectories, registered_md_failure_reason, torchsim_md_succeeded
from metric_sources import cohort_trajectory_path
from model_display_names import display_model_name, normalize_display_key


SYSTEM = "bulkCu_1000K_Kapil"
ASE_SOURCE = "mlip-trajs-ase-accelerated"
TORCHSIM_SOURCE = "mlip-trajs-torchsim-accelerated"


class AseMDInputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.failures = {}
        registry = patch("md_success.load_failed_md_runs", return_value=self.failures)
        registry.start()
        self.addCleanup(registry.stop)

    def run_output(self, source, model, *, calculator=None, system=SYSTEM,
                   timing_system=None, steps=100):
        directory = self.root / source / system
        directory.mkdir(parents=True, exist_ok=True)
        trajectory = directory / f"nvt_{model}.h5"
        trajectory.touch()
        with trajectory.with_name(f"md_timing_{model}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["calculator", "system", "n_steps"])
            writer.writeheader()
            writer.writerow({"calculator": model if calculator is None else calculator,
                             "system": system if timing_system is None else timing_system,
                             "n_steps": steps})
        return trajectory

    def test_model_aliases_remove_runner_tags_but_preserve_execution_and_checkpoints(self):
        aliases = {
            "nequip-oam-l-force-only-ase": "nequip",
            "CHGNet-force-only-ase": "chgnet",
            "mattersim-v1-5M-compile-force-only-ase": "mattersim-v1-5m-compile",
            "mace-mp-0-compile-ase": "mace-mp-0-compile",
            "uma-s-omat-compile-force-only-ase": "uma-s-omat-compile",
            "uma-s-omat-turbo-force-only-ase": "uma-s-omat-turbo",
            "pet-oam-xl-force-only-torchscript-ase": "pet-oam-xl-torchscript",
            "orb-v3-omat-stress-ase": "orb-v3-omat",
            "orb-v3-direct-omat-force-only-ase": "orb-v3-direct-omat",
            "orb-v2-force-only-eager-ase": "orb-v2",
        }
        for raw, expected in aliases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_display_key(raw), expected)
        self.assertNotEqual(normalize_display_key("uma-s-omat-compile-ase"),
                            normalize_display_key("uma-s-omat-turbo-ase"))
        self.assertNotEqual(normalize_display_key("orb-v3-omat-ase"),
                            normalize_display_key("orb-v3-direct-omat-ase"))
        self.assertEqual(display_model_name("nequip-oam-l-force-only-ase"), "NequIP-OAM-L")
        self.assertEqual(display_model_name("uma-s-omat-turbo-force-only-ase"), "UMA-S-P1 (turbo)")

    def test_ase_completion_accepts_runner_tag_in_supported_physical_layouts(self):
        for source in (ASE_SOURCE, "mlip-trajs-ase", "mlip-trajs-ase-eager",
                       "ase/md_accelerated", "ase/md_eager"):
            for model, calculator in (("nequip-oam-l-force-only-ase", "nequip-oam-l-force-only"),
                                      ("nequip-oam-l-force-only", "nequip-oam-l-force-only-ase"),
                                      ("nequip-oam-l-force-only-ase", "nequip-oam-l-force-only-ase")):
                with self.subTest(source=source, calculator=calculator):
                    trajectory = self.run_output(source, model, calculator=calculator)
                    self.assertTrue(torchsim_md_succeeded(trajectory))

    def test_torchsim_and_unidentified_sources_still_require_exact_raw_models(self):
        for source in (TORCHSIM_SOURCE, "mlip-trajs-torchsim-eager",
                       "torchsim/md_accelerated", "unidentified"):
            with self.subTest(source=source):
                trajectory = self.run_output(source, "nequip-oam-l-force-only-ase",
                                             calculator="nequip-oam-l-force-only")
                self.assertFalse(torchsim_md_succeeded(trajectory))
                trajectory = self.run_output(source, "nequip-oam-l-force-only-ase")
                self.assertTrue(torchsim_md_succeeded(trajectory))

    def test_runner_alias_does_not_authorize_other_models_properties_or_systems(self):
        model = "uma-s-omat-turbo-force-only-ase"
        for calculator in ("uma-s-omat-compile-force-only", "uma-m-omat-turbo-force-only",
                           "uma-s-omat-turbo-stress", "UMA-S-omat-turbo-force-only"):
            with self.subTest(calculator=calculator):
                trajectory = self.run_output(ASE_SOURCE, model, calculator=calculator)
                self.assertFalse(torchsim_md_succeeded(trajectory))
        trajectory = self.run_output(ASE_SOURCE, model, calculator=model.removesuffix("-ase"),
                                     timing_system="different_system")
        self.assertFalse(torchsim_md_succeeded(trajectory))

    def test_missing_and_failed_markers_remain_failures(self):
        model = "orb-v2-force-only-ase"
        trajectory = self.run_output(ASE_SOURCE, model, calculator="orb-v2-force-only", steps=0)
        self.assertFalse(torchsim_md_succeeded(trajectory))
        timing = trajectory.with_name(f"md_timing_{model}.csv")
        timing.write_text("calculator,system,n_steps\n")
        self.assertFalse(torchsim_md_succeeded(trajectory))
        timing.unlink()
        self.assertFalse(torchsim_md_succeeded(trajectory))
        trajectory = self.run_output(ASE_SOURCE, model, calculator="orb-v2-force-only")
        trajectory.unlink()
        self.assertFalse(torchsim_md_succeeded(trajectory))

    def test_audited_failure_exclusions_stay_scoped_to_physical_source_and_raw_model(self):
        model = "orb-v2-force-only-ase"
        self.failures[(ASE_SOURCE, SYSTEM, model)] = "Audited physical failure"
        failed = self.run_output(ASE_SOURCE, model, calculator="orb-v2-force-only")
        self.assertFalse(torchsim_md_succeeded(failed))
        other = self.run_output("mlip-trajs-ase-eager", model, calculator="orb-v2-force-only")
        self.assertTrue(torchsim_md_succeeded(other))
        other = self.run_output(ASE_SOURCE, "orb-v3-direct-omat-force-only-ase",
                                calculator="orb-v3-direct-omat-force-only")
        self.assertTrue(torchsim_md_succeeded(other))
        self.failures[("mlip-trajs-ase-eager", SYSTEM, model)] = "Audited eager failure"
        self.assertFalse(torchsim_md_succeeded(
            self.root / "mlip-trajs-ase-eager" / SYSTEM / f"nvt_{model}.h5"))

    def test_canonical_model_resolves_native_ase_trajectory_with_real_runner_tag(self):
        trajectory = self.run_output(ASE_SOURCE, "nequip-oam-l-force-only-ase",
                                     calculator="nequip-oam-l-force-only")
        self.assertEqual(cohort_trajectory_path(self.root, ASE_SOURCE, SYSTEM, "nequip"), trajectory)
        self.assertTrue(torchsim_md_succeeded(trajectory))
        for variant in ("compile", "turbo"):
            run = self.run_output(ASE_SOURCE, f"uma-s-omat-{variant}-force-only-ase",
                                  calculator=f"uma-s-omat-{variant}-force-only")
            self.assertEqual(cohort_trajectory_path(self.root, ASE_SOURCE, SYSTEM,
                                                    f"uma-s-omat-{variant}"), run)

    def test_canonical_ase_eager_source_resolves_physical_eager_directory(self):
        completed = self.run_output("mlip-trajs-ase-eager", "chgnet-force-only-ase",
                                    calculator="chgnet-force-only")
        runs = discover_md_trajectories(self.root / "mlip-trajs-ase")
        self.assertEqual(runs, {SYSTEM: {"chgnet": completed}})
        self.assertTrue(torchsim_md_succeeded(runs[SYSTEM]["chgnet"]))

    def test_ase_discovery_collapses_stale_alias_and_keeps_one_model_across_systems(self):
        model = "orb-v3-direct-omat-force-only"
        completed = self.run_output(ASE_SOURCE, model)
        stale = self.run_output(ASE_SOURCE, model + "-ase", calculator=model, steps=0)
        stale.unlink()
        other_system = "bulkAg_600K_Kapil"
        other = self.run_output(ASE_SOURCE, model + "-ase", calculator=model,
                                system=other_system)
        runs = discover_md_trajectories(self.root / ASE_SOURCE)
        self.assertEqual(runs, {SYSTEM: {"orb-v3-direct-omat": completed},
                                other_system: {"orb-v3-direct-omat": other}})
        self.assertTrue(all(torchsim_md_succeeded(path) for models in runs.values()
                            for path in models.values()))

    def test_two_completed_ase_aliases_require_disambiguation(self):
        model = "orb-v3-direct-omat-force-only"
        first = self.run_output(ASE_SOURCE, model)
        second = self.run_output(ASE_SOURCE, model + "-ase", calculator=model)
        with self.assertRaisesRegex(ValueError, "Ambiguous completed ASE MD runs") as raised:
            discover_md_trajectories(self.root / ASE_SOURCE)
        self.assertIn(str(first), str(raised.exception))
        self.assertIn(str(second), str(raised.exception))
        self.assertIn(SYSTEM, str(raised.exception))
        self.assertIn("orb-v3-direct-omat", str(raised.exception))

    def test_all_failed_ase_aliases_retain_original_audited_failure_path(self):
        model = "orb-v3-direct-omat-force-only"
        audited = self.run_output(ASE_SOURCE, model + "-ase", calculator=model)
        self.failures[(ASE_SOURCE, SYSTEM, model + "-ase")] = "Audited unphysical run"
        self.run_output(ASE_SOURCE, model, steps=0)
        runs = discover_md_trajectories(self.root / ASE_SOURCE)
        selected = runs[SYSTEM]["orb-v3-direct-omat"]
        self.assertEqual(selected, audited)
        self.assertEqual(registered_md_failure_reason(selected), "Audited unphysical run")
        self.assertFalse(torchsim_md_succeeded(selected))

    def test_failed_ase_discovery_prefers_existing_h5_and_retains_execution_variants(self):
        model = "orb-v2-force-only"
        partial = self.run_output(ASE_SOURCE, model + "-ase", calculator=model, steps=0)
        missing = self.run_output(ASE_SOURCE, model, steps=0)
        missing.unlink()
        compile_run = self.run_output(ASE_SOURCE, "uma-s-omat-compile-force-only-ase",
                                      calculator="uma-s-omat-compile-force-only")
        turbo_run = self.run_output(ASE_SOURCE, "uma-s-omat-turbo-force-only-ase",
                                    calculator="uma-s-omat-turbo-force-only")
        self.assertEqual(discover_md_trajectories(self.root / ASE_SOURCE)[SYSTEM],
                         {"orb-v2": partial, "uma-s-omat-compile": compile_run,
                          "uma-s-omat-turbo": turbo_run})

    def test_torchsim_discovery_keeps_raw_alias_keys_and_physical_paths(self):
        model = "nequip-oam-l-force-only"
        first = self.run_output(TORCHSIM_SOURCE, model)
        second = self.run_output(TORCHSIM_SOURCE, model + "-ase")
        runs = discover_md_trajectories(self.root / TORCHSIM_SOURCE)
        self.assertEqual(runs, {SYSTEM: {model: first, model + "-ase": second}})


if __name__ == "__main__":
    unittest.main()
