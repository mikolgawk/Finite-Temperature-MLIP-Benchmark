"""Validate native runner timing comparisons without creating any figures."""

import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import figure_SI_17 as timing
import plot_all


METAL = "bulkCu_1000K_Kapil"
CRYSTAL = "naphthalene_295K_Sharma_S"


class FigureSI17Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "data"
        for source in (timing.TORCHSIM_SOURCE, timing.ASE_SOURCE):
            (self.data / source).mkdir(parents=True)
        failures = {}
        for target in ("figure_SI_17.load_failed_md_runs", "md_success.load_failed_md_runs"):
            patcher = patch(target, return_value=failures)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.failures = failures

    def add_run(self, source, model="orb-v2-force-only", system=METAL,
                seconds=0.001, steps=100, row_fields=None, **fields):
        folder = self.data / source / system
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"md_timing_{model}.csv"
        row = {
            "calculator": model.removesuffix("-ase"), "system": system,
            "n_steps": steps, "elapsed_seconds": seconds * steps,
            "seconds_per_step": seconds, "time_step_fs": 1, "record_interval": 1,
            "engine": source,
        }
        row.update(fields)
        row.update(row_fields or {})
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        path.with_name(f"nvt_{model}.h5").touch()
        return path

    def observations(self, source, include_crystals=False):
        return timing.load_native_timings(self.data / source, source, include_crystals)[0]

    def compare(self):
        return timing.compare_timings(self.observations(timing.TORCHSIM_SOURCE),
                                      self.observations(timing.ASE_SOURCE))

    def test_backend_aliases_match_but_execution_variants_remain_distinct(self):
        aliases = {
            "MatterSim-v1-5M-compile-force-only-ase": "mattersim-v1-5m-compile",
            "nequip-oam-l-force-only-ase": "nequip",
            "orb-v3-omat-force-only": "orb-v3-omat",
            "orb-v3-direct-omat-force-only-ase": "orb-v3-direct-omat",
            "uma-s-omat-turbo-force-only-ase": "uma-s-omat-turbo",
            "pet-oam-xl-force-only-torchscript-ase": "pet-oam-xl-torchscript",
            "mace-mp-0-compile-ase": "mace-mp-0-compile",
        }
        for raw, canonical in aliases.items():
            self.assertEqual(timing.timing_model_key(raw), canonical)
        self.add_run(timing.TORCHSIM_SOURCE)
        self.add_run(timing.ASE_SOURCE, "orb-v2-force-only-ase")
        self.add_run(timing.TORCHSIM_SOURCE, "mace-mp-0-compile")
        self.add_run(timing.ASE_SOURCE, "mace-mp-0")
        self.add_run(timing.TORCHSIM_SOURCE, "orb-v3-omat-force-only")
        self.add_run(timing.ASE_SOURCE, "orb-v3-direct-omat-force-only")
        paired, summary = self.compare()
        self.assertEqual(paired.model.tolist(), ["orb-v2"])
        self.assertEqual(summary.model_display.tolist(), ["orb-v2"])
        self.assertEqual(paired.raw_model_ase.iloc[0], "orb-v2-force-only-ase")

    def test_pet_torchscript_names_match_using_engine_metadata(self):
        for model in ("pet-oam-xl", "pet-omat-xl"):
            torchsim_model = model + "-force-only-torchscript"
            for system in (METAL, "unconfirmed"):
                self.add_run(timing.TORCHSIM_SOURCE, torchsim_model, system=system,
                             seconds=0.002, engine="torch-sim-0.6.1+pet-torchscript-force-only")
            self.add_run(timing.ASE_SOURCE, model + "-ase", seconds=0.003,
                         engine="ase+upet-0.2.6-torchscript-force-only")
            self.add_run(timing.ASE_SOURCE, model, system="unconfirmed", engine="")
            self.assertEqual(timing.timing_model_key(model, "ASE+TorchScript"),
                             model + "-torchscript")
            for engine in ("", "ase+upet-eager", None):
                self.assertEqual(timing.timing_model_key(model, engine), model)
            self.assertEqual(timing.timing_model_key(torchsim_model), model + "-torchscript")
        paired, summary = self.compare()
        self.assertEqual(paired.model.tolist(), ["pet-oam-xl-torchscript", "pet-omat-xl-torchscript"])
        self.assertEqual(paired.system.tolist(), [METAL, METAL])
        self.assertEqual(summary.model_display.tolist(), ["PET-OAM-XL", "PET-OMAT-XL"])
        self.assertEqual(summary.n_common_systems.tolist(), [1, 1])
        self.assertEqual(summary.torchsim_mean_ms_per_step.tolist(), [2.0, 2.0])
        self.assertEqual(summary.ase_mean_ms_per_step.tolist(), [3.0, 3.0])
        self.assertEqual(summary.torchsim_speedup.tolist(), [1.5, 1.5])

    def test_ambiguous_pet_torchscript_aliases_are_rejected(self):
        for model in ("pet-oam-xl", "pet-oam-xl-force-only-torchscript-ase"):
            self.add_run(timing.ASE_SOURCE, model, engine="ase+upet-0.2.6-torchscript-force-only")
        with self.assertRaisesRegex(ValueError, "Ambiguous timing files"):
            self.observations(timing.ASE_SOURCE)

    def test_failed_partial_and_crystal_runs_are_excluded(self):
        source = timing.TORCHSIM_SOURCE
        self.add_run(source)
        self.add_run(source, system=CRYSTAL)
        partial = self.add_run(source, system="partial")
        partial.write_text("calculator,system,n_steps\n")
        missing = self.add_run(source, system="missing")
        missing.with_name("nvt_orb-v2-force-only.h5").unlink()
        self.add_run(source, system="unphysical")
        self.failures[(source, "unphysical", "orb-v2-force-only")] = "Audited physical failure"
        frame, skipped = timing.load_native_timings(self.data / source, source)
        self.assertEqual(frame.system.tolist(), [METAL])
        self.assertEqual(len(skipped), 3)
        self.assertTrue(any("Audited physical failure" in reason for reason in skipped))
        self.assertEqual(set(self.observations(source, True).system), {METAL, CRYSTAL})

    def test_invalid_timing_records_cannot_enter_the_comparison(self):
        source = timing.TORCHSIM_SOURCE
        self.add_run(source)
        invalid_fields = [
            {"seconds_per_step": 0}, {"seconds_per_step": float("nan")},
            {"n_steps": 3.5}, {"elapsed_seconds": float("inf")},
            {"elapsed_seconds": 999}, {"calculator": "different-model"},
            {"system": "different-system"}, {"record_interval": 0},
            {"time_step_fs": float("inf")},
        ]
        for index, fields in enumerate(invalid_fields):
            self.add_run(source, system=f"invalid_{index}", row_fields=fields)
        frame, skipped = timing.load_native_timings(self.data / source, source)
        self.assertEqual(frame.system.tolist(), [METAL])
        self.assertEqual(len(skipped), len(invalid_fields))

    def test_same_system_means_are_equal_weighted_and_never_borrow_eager_runs(self):
        for source, values in ((timing.TORCHSIM_SOURCE, [0.001, 0.003]),
                               (timing.ASE_SOURCE, [0.006, 0.002])):
            for system, seconds, steps in zip((METAL, "Pt111w24H2O_380K_Heenen_VASP"), values, (10, 100)):
                self.add_run(source, system=system, seconds=seconds, steps=steps)
        self.add_run(timing.TORCHSIM_SOURCE, system="unmatched", seconds=99)
        for model in ("chgnet", "esen-30m-oam", "eq-v2-m-omat"):
            self.add_run("mlip-trajs-torchsim-eager", model)
            self.add_run("mlip-trajs-ase-eager", model)
            self.add_run(timing.ASE_SOURCE, model)
        paired, summary = self.compare()
        self.assertEqual(set(paired.system), {METAL, "Pt111w24H2O_380K_Heenen_VASP"})
        self.assertEqual(summary.model.tolist(), ["orb-v2"])
        row = summary.iloc[0]
        self.assertEqual(row.n_common_systems, 2)
        self.assertEqual(row.torchsim_mean_ms_per_step, 2)
        self.assertEqual(row.ase_mean_ms_per_step, 4)
        self.assertEqual(row.torchsim_speedup, 2)

    def test_workload_mismatches_are_dropped_and_empty_inputs_are_supported(self):
        for source in (timing.TORCHSIM_SOURCE, timing.ASE_SOURCE):
            for system in ("steps", "timestep", "interval", "legacy"):
                fields = {}
                if source == timing.ASE_SOURCE:
                    fields = {"steps": {"n_steps": 50, "elapsed_seconds": 0.05},
                              "timestep": {"time_step_fs": 2},
                              "interval": {"record_interval": 2},
                              "legacy": {"time_step_fs": "", "record_interval": ""}}[system]
                self.add_run(source, system=system, **fields)
        paired, summary = self.compare()
        self.assertEqual(paired.system.tolist(), ["legacy"])
        self.assertEqual(summary.n_common_systems.tolist(), [1])
        empty = self.observations(timing.ASE_SOURCE).iloc[:0]
        for left, right in ((empty, empty), (empty, self.observations(timing.TORCHSIM_SOURCE))):
            pairs, means = timing.compare_timings(left, right)
            self.assertTrue(pairs.empty)
            self.assertTrue(means.empty)
            self.assertEqual(means.columns.tolist(), timing.SUMMARY_COLUMNS)
        self.assertFalse(timing.comparison_available(self.root / "absent"))

    def test_duplicate_normalized_model_system_files_are_rejected(self):
        self.add_run(timing.ASE_SOURCE, "orb-v2-force-only-ase")
        self.add_run(timing.ASE_SOURCE, "orb-v2-stress-ase")
        with self.assertRaisesRegex(ValueError, "Ambiguous timing files"):
            self.observations(timing.ASE_SOURCE)

    def test_finished_benchmark_can_include_terminal_failed_systems(self):
        source = timing.ASE_SOURCE
        raw_model = "orb-v2-force-only-ase"
        self.add_run(source, raw_model)
        failed = self.add_run(source, raw_model, system="exception")
        failed.write_bytes(b"")
        trajectory = failed.with_name(f"nvt_{raw_model}.h5")
        trajectory.rename(trajectory.with_name(trajectory.name + ".inprogress"))
        self.failures[(source, "audited", raw_model)] = "Audited physical failure"

        finished = timing.finished_model_keys(
            self.data / source, source, self.observations(source),
            expected_systems={METAL, "exception", "audited"},
        )

        self.assertEqual(finished, {"orb-v2"})
        # Failed attempts establish a terminal result; they do not add timings.
        self.assertEqual(self.observations(source).system.tolist(), [METAL])

    def test_unfinished_systems_prevent_benchmark_completion(self):
        source = timing.ASE_SOURCE
        cases = (
            ("orb-v2-force-only-ase", "missing"),
            ("uma-s-omat-turbo-force-only-ase", "inprogress"),
            ("mace-mp-0-compile-ase", "header-only"),
            ("mace-mpa-0-compile-ase", "invalid-record"),
        )
        for raw_model, status in cases:
            with self.subTest(status=status):
                self.add_run(source, raw_model)
                if status == "inprogress":
                    folder = self.data / source / "unfinished"
                    folder.mkdir(exist_ok=True)
                    (folder / f"nvt_{raw_model}.h5.inprogress").touch()
                elif status in {"header-only", "invalid-record"}:
                    path = self.add_run(
                        source, raw_model, system="unfinished",
                        row_fields={"n_steps": 0} if status == "invalid-record" else None,
                    )
                    if status == "header-only":
                        path.write_text("calculator,system,n_steps\n")

                finished = timing.finished_model_keys(
                    self.data / source, source, self.observations(source),
                    expected_systems={METAL, "unfinished"},
                )

                self.assertNotIn(timing.timing_model_key(raw_model), finished)

    def test_failed_pet_aliases_use_confirmed_torchscript_configuration(self):
        source = timing.ASE_SOURCE
        for model in ("pet-oam-xl", "pet-omat-xl"):
            raw_model = model + "-ase"
            self.add_run(source, raw_model, engine="ase+upet-0.2.6-torchscript-force-only")
            failed = self.add_run(source, raw_model, system="H_1050K_Rupp_QE")
            failed.write_bytes(b"")
            failed.with_name(f"nvt_{raw_model}.h5").unlink()

        finished = timing.finished_model_keys(
            self.data / source, source, self.observations(source),
            expected_systems={METAL, "H_1050K_Rupp_QE"},
        )

        self.assertEqual(finished, {"pet-oam-xl-torchscript", "pet-omat-xl-torchscript"})

    def test_data_only_cli_exports_tables_without_calling_plotting(self):
        self.add_run(timing.TORCHSIM_SOURCE)
        self.add_run(timing.ASE_SOURCE, "orb-v2-force-only-ase")
        output = self.root / "never-created.pdf"
        summary, paired = self.root / "summary.csv", self.root / "paired.csv"
        metadata = self.root / "metadata.json"
        metadata.write_text(json.dumps({METAL: {}}))
        argv = ["figure_SI_17.py", "--data-only",
                "--torchsim-dir", str(self.data / timing.TORCHSIM_SOURCE),
                "--ase-dir", str(self.data / timing.ASE_SOURCE),
                "--output-file", str(output), "--output-csv", str(summary),
                "--paired-csv", str(paired), "--metadata-file", str(metadata)]
        with (patch.object(sys, "argv", argv),
              patch.object(timing, "plot_comparison", side_effect=AssertionError("Must not plot")) as plot,
              contextlib.redirect_stdout(io.StringIO())):
            timing.main()
        plot.assert_not_called()
        self.assertTrue(summary.is_file())
        self.assertTrue(paired.is_file())
        self.assertFalse(output.exists())

    def test_pipeline_includes_comparison_once_and_respects_output_paths(self):
        argv = ["plot_all.py", "--dry-run", "--include-molecular-crystals",
                "--source", timing.TORCHSIM_SOURCE, "--source", "mlip-trajs-torchsim-eager",
                "--plots-dir", str(self.root / "plots"), "--results-dir", str(self.root / "results")]
        with (patch.object(sys, "argv", argv),
              patch.object(plot_all, "run_stage", return_value=True) as stage,
              contextlib.redirect_stdout(io.StringIO())):
            plot_all.main()
        commands = [call.args[1] for call in stage.call_args_list
                    if Path(call.args[1][2]).name == "figure_SI_17.py"]
        self.assertEqual(len(commands), 1)
        command = commands[0]
        self.assertIn("--include-molecular-crystals", command)
        self.assertEqual(command[command.index("--output-file") + 1],
                         str(self.root / "plots" / timing.COMPARISON_DIR / "figure_SI_17.pdf"))
        self.assertEqual(command[command.index("--output-csv") + 1],
                         str(self.root / "results" / timing.COMPARISON_DIR / "figure_SI_17.csv"))
        self.assertEqual(command[command.index("--ase-dir") + 1],
                         str(plot_all.CONFIG_DIR / "data" / timing.ASE_SOURCE))
        self.assertFalse((self.root / "plots").exists())
        with (patch.object(sys, "argv", ["plot_all.py", "--source", timing.TORCHSIM_SOURCE]),
              patch.object(plot_all, "comparison_available", return_value=False),
              patch.object(plot_all, "require_vdos_metrics_file"),
              patch.object(plot_all, "run_stage", return_value=True) as stage,
              contextlib.redirect_stdout(io.StringIO())):
            plot_all.main()
        self.assertFalse(any(Path(call.args[1][2]).name == "figure_SI_17.py"
                             for call in stage.call_args_list))


if __name__ == "__main__":
    unittest.main()
