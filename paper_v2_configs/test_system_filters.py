"""Check that default system selection affects computations and timing means."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import write

from system_filters import (
    filter_molecular_crystals, is_molecular_crystal,
    filter_pressure_systems, include_pressure_system, include_system,
    is_metal_water_interface,
)


HERE = Path(__file__).resolve().parent
SOURCE = "mlip-trajs-torchsim-eager"
METAL = "bulkCu_1000K_Kapil"
CRYSTAL = "naphthalene_295K_Sharma_S"
INTERFACE = "Pt111w24H2O_380K_Heenen_VASP"


def load_module(relative):
    path = HERE / relative
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("filter_test_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SystemFilterTests(unittest.TestCase):
    def test_pressure_exclusion_is_specific_to_the_interface(self):
        for value in ("Pt111w24H2O", INTERFACE, f"/data/{INTERFACE}/nvt_model.h5",
                      "Metal-water interfaces"):
            self.assertTrue(is_metal_water_interface(value))
            self.assertFalse(include_pressure_system(value, True))
        alloy = "bulkPt3Co_300K_J.Kioseoglou_VASP"
        self.assertTrue(include_pressure_system(alloy))
        self.assertTrue(include_system(INTERFACE))
        frame = pd.DataFrame({"system": [METAL, alloy, INTERFACE, CRYSTAL]})
        self.assertEqual(filter_pressure_systems(frame).system.tolist(), [METAL, alloy])
        self.assertEqual(filter_pressure_systems(frame, True).system.tolist(),
                         [METAL, alloy, CRYSTAL])
        for column, value in (("system_type", "Metal-water interfaces"),
                              ("trajectory_file", f"/ref/{INTERFACE}/traj.extxyz")):
            self.assertTrue(filter_pressure_systems(pd.DataFrame({column: [value]})).empty)
        self.assertEqual(len(frame), 4)
        self.assertTrue(filter_pressure_systems(frame.iloc[:0]).empty)

    def test_pressure_correlations_filter_old_detailed_tables(self):
        frame = pd.DataFrame({"model": ["mace-mp-0"] * 2,
                              "system": [METAL, INTERFACE],
                              "pressure_error_percent": [20.0, 100.0]})
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "pressure.csv"
            frame.to_csv(csv, index=False)
            tables = load_module("pareto_plots/_figure_data.py")
            self.assertEqual(tables.load_pressure(csv).pressure_error_percent.tolist(), [20.0])
        correlations = load_module("vdos/figure_SI_7_8_9.py")
        self.assertEqual(correlations.standardize_pressure(frame).pressure_error_percent.tolist(),
                         [20.0])
        for script in ("pareto_plots/figure_7.py", "pareto_plots/figure_SI_13.py"):
            figure = load_module(script)
            prepared = figure._prepare_pressure_df(frame, 10.0, True)
            self.assertEqual(prepared["Pressure Error [%]"].tolist(), [20.0])

    def test_names_paths_and_saved_tables(self):
        for name in ("anthracene", "naphthalene", "pentacene", "picene", "tetracene"):
            self.assertTrue(is_molecular_crystal(name))
            self.assertTrue(
                is_molecular_crystal(f"/ref/{name}_295K_Sharma_S/traj.extxyz")
            )
        frame = pd.DataFrame({"system": [METAL, CRYSTAL], "error": [1.0, 99.0]})
        self.assertEqual(filter_molecular_crystals(frame).system.tolist(), [METAL])
        self.assertEqual(len(filter_molecular_crystals(frame, True)), 2)
        self.assertEqual(len(frame), 2)
        self.assertTrue(filter_molecular_crystals(frame.iloc[:0]).empty)

    @contextlib.contextmanager
    def trajectories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = {}
            for system, seconds in ((METAL, 1.0), (CRYSTAL, 9.0)):
                ref = root / "data/ref-trajs" / system / "traj.extxyz"
                ref.parent.mkdir(parents=True)
                frames = []
                for index in range(8):
                    frames.append(
                        Atoms(
                            "Cu2",
                            positions=[[0, 0, 0], [1 + 0.05 * np.sin(index), 0, 0]],
                            cell=[6, 6, 6],
                            pbc=True,
                        )
                    )
                write(ref, frames)
                folder = root / "data" / SOURCE / system
                folder.mkdir(parents=True)
                with h5py.File(folder / "nvt_mace-mp-0.h5", "w") as handle:
                    handle["data/positions"] = np.stack([a.positions for a in frames])
                    handle["data/cell"] = np.stack([a.cell.array.T for a in frames])
                    handle["data/atomic_numbers"] = [[29, 29]]
                    handle["data/velocities"] = np.stack(
                        [np.full((2, 3), np.cos(index)) for index in range(8)]
                    )
                    handle["data/pbc"] = [True, True, True]
                pd.DataFrame(
                    [
                        dict(
                            calculator="mace-mp-0",
                            system=system,
                            n_steps=7,
                            elapsed_seconds=7 * seconds,
                            seconds_per_step=seconds,
                        )
                    ]
                ).to_csv(folder / "md_timing_mace-mp-0.csv", index=False)
                metadata[system] = dict(
                    n_frames=8,
                    timestep=1.0,
                    timestep_units="fs",
                    trajectory_length_ps=0.007,
                    position_print_stride=1,
                )
            (root / "data/ref-trajs/md_metadata.json").write_text(json.dumps(metadata))
            yield root, metadata

    def test_standalone_rdf_filters_before_loading_and_scoring(self):
        rdf = load_module(
            "rdfs/get-rdf-and-results-by-system-type-same-simulation-length.py"
        )
        with self.trajectories() as (root, _):
            for include, expected in ((False, {METAL}), (True, {METAL, CRYSTAL})):
                argv = [
                    "rdf",
                    "--source",
                    SOURCE,
                    "--results-dir",
                    str(root / "results"),
                ]
                if include:
                    argv.append("--include-molecular-crystals")
                with (
                    patch.object(sys, "argv", argv),
                    patch.object(rdf, "REF_TRAJ_BASE_DIR", root / "data/ref-trajs"),
                    patch.dict(rdf.MLIP_TRAJ_DIRS, {SOURCE: root / "data" / SOURCE}),
                    patch.object(
                        rdf,
                        "load_reference_trajectory",
                        wraps=rdf.load_reference_trajectory,
                    ) as load,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    rdf.main()
                self.assertEqual(
                    {call.args[0].parent.name for call in load.call_args_list}, expected
                )
                rows = pd.read_csv(
                    root
                    / "results"
                    / SOURCE
                    / "rdf_similarity_scores_same_simulation_length_mace-mp-0.csv"
                )
                self.assertEqual(set(rows.System), expected)

    def test_standalone_vdos_filters_computations_and_outputs(self):
        vdos = load_module("vdos/get_normalized_VDOS.py")
        with self.trajectories() as (root, metadata):
            for include, expected in ((False, {METAL}), (True, {METAL, CRYSTAL})):
                argv = [
                    "vdos",
                    "--source",
                    SOURCE,
                    "--data-dir",
                    str(root / "data"),
                    "--results-dir",
                    str(root / "results"),
                ]
                if include:
                    argv.append("--include-molecular-crystals")
                with (
                    patch.object(sys, "argv", argv),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    args = vdos.parse_args()
                    vdos.process_source(
                        args,
                        SOURCE,
                        data_dir=root / "data",
                        reference_dir=root / "data/ref-trajs",
                        metadata=metadata,
                    )
                rows = pd.read_csv(
                    root
                    / "results"
                    / SOURCE
                    / "vdos_pair_errors_ev_normalized_same_simulation_length.csv"
                )
                self.assertEqual(set(rows.system), expected)

    def test_pressure_histograms_skip_crystals_before_scoring(self):
        pressure = load_module("pressures/get_model_pressure_errors.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "references").mkdir()
            frames = pd.DataFrame(
                {
                    "trajectory_file": [
                        f"/ref/{s}/traj.extxyz"
                        for s in (METAL, CRYSTAL, INTERFACE)
                        for _ in range(3)
                    ],
                    "frame_index": [0, 1, 2] * 3,
                    "pressure_GPa": [1.0, 2.0, 3.0] * 3,
                }
            )
            frames.to_csv(
                root / ("mace-mp-0" + pressure.DEFAULT_MODEL_FILE_SUFFIX), index=False
            )
            frames.to_csv(root / "references/mace-mp-0.csv", index=False)
            for exclusions, expected in ((None, {METAL}), (set(), {METAL, CRYSTAL})):
                kwargs = (
                    {} if exclusions is None else {"excluded_system_types": exclusions}
                )
                with patch.object(
                    pressure,
                    "pressure_histogram_similarity",
                    wraps=pressure.pressure_histogram_similarity,
                ) as score:
                    rows = pressure.build_pair_rows(
                        root, None, pressure.DEFAULT_MODEL_FILE_SUFFIX, 20, **kwargs
                    )
                self.assertEqual(set(rows.system), expected)
                self.assertEqual(score.call_count, len(expected))

    def test_every_timing_loader_excludes_crystals_from_averages(self):
        paths = [
            "pareto_plots/plot-model-timings.py",
            "pareto_plots/figure_7.py",
            "pareto_plots/figure_SI_13.py",
            "rdfs/figure_SI_14.py",
            "vdos/figure_SI_16.py",
            "pressures/figure_SI_15.py",
        ]
        with self.trajectories() as (root, _):
            timing_root = root / "data" / SOURCE
            for path in paths:
                with self.subTest(script=path):
                    module = load_module(path)
                    if hasattr(module, "read_timing_files"):
                        default, _ = module.read_timing_files(timing_root)
                        included, _ = module.read_timing_files(
                            timing_root, include_molecular_crystals=True
                        )
                        self.assertEqual(default.system.tolist(), [METAL])
                        self.assertEqual(set(included.system), {METAL, CRYSTAL})
                    else:
                        default = module.load_model_avg_timings(timing_root)
                        included = module.load_model_avg_timings(
                            timing_root, include_molecular_crystals=True
                        )
                        column = next(c for c in default if c.startswith("mean_time"))
                        self.assertAlmostEqual(default[column].iloc[0], 1000.0)
                        self.assertAlmostEqual(included[column].iloc[0], 5000.0)

    def test_pressure_evaluation_defaults_and_cached_system_selection(self):
        evaluator = load_module("pressures/pressure_evaluator.py")
        with self.trajectories() as (root, _):
            # An excluded interface must never need valid frames or reference metadata.
            interface_folder = root / "data" / SOURCE / INTERFACE
            interface_folder.mkdir()
            (interface_folder / "nvt_mace-mp-0.h5").touch()
            (interface_folder / "md_timing_mace-mp-0.csv").write_text(
                f"calculator,system,n_steps\nmace-mp-0,{INTERFACE},7\n"
            )
            # CuAu has no reference stress and should not invalidate the cache.
            skipped = "bulkCuAu_500K-Artrith_VASP"
            folder = root / "data" / SOURCE / skipped
            folder.mkdir()
            (folder / "nvt_mace-mp-0.h5").touch()
            (folder / "md_timing_mace-mp-0.csv").write_text(
                f"calculator,system,n_steps\nmace-mp-0,{skipped},7\n"
            )
            metadata_path = root / "data/ref-trajs/md_metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata[skipped] = {"stress_print_stride": None}
            metadata_path.write_text(json.dumps(metadata))
            args = SimpleNamespace(
                traj_dir=root / "data" / SOURCE,
                ref_dir=root / "data/ref-trajs",
                metadata=root / "data/ref-trajs/md_metadata.json",
                output_dir=root / "results",
                trajectory_model=None,
                systems=None,
                max_frames=None,
                force=False,
                no_progress=True,
                debug=False,
                include_molecular_crystals=False,
            )

            def reference(path, meta):
                return pd.DataFrame(
                    {
                        "system": [path.parent.name] * 8,
                        "trajectory_file": [str(path)] * 8,
                        "frame_index": np.arange(8),
                        "time_fs": np.arange(8),
                        "pressure_GPa": np.zeros(8),
                    }
                )

            for include, expected, n_calls in (
                (False, {METAL}, 8),
                (False, {METAL}, 0),
                (True, {METAL, CRYSTAL}, 16),
                (False, {METAL}, 8),
            ):
                args.include_molecular_crystals = include
                predict = Mock(side_effect=lambda atoms, state: (np.zeros(6), state))
                with (
                    patch.object(evaluator, "_ARGS", args),
                    patch.object(evaluator, "_BACKEND", "torchsim"),
                    patch.object(
                        evaluator, "_reference_pressures", side_effect=reference
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    evaluator._run("mace-mp-0", predict, "test")
                summary = pd.read_csv(
                    root
                    / "results"
                    / "mace-mp-0_same-simulation-length_pressure_trajectory_summary.csv"
                )
                self.assertEqual(set(summary.system), expected)
                self.assertEqual(predict.call_count, n_calls)


if __name__ == "__main__":
    unittest.main()
