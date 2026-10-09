"""CPU regression tests; no MLIP checkpoints or GPU required."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from argparse import Namespace
import h5py
import numpy as np
import pandas as pd
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
import pressure_pipeline as pipeline
from get_model_pressure_errors import (
    add_failed_system_penalties,
    build_pair_rows,
    build_pressure_mae_from_trajectory_summaries,
    compute_pressure_metric,
    resolve_reference_pressure_file,
    write_metric_outputs,
)

HERE = Path(__file__).resolve().parent


class PressurePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.structure = "bulkCu_300K_test"
        self.meta = {"timestep": 2., "position_print_stride": 2}

    def tearDown(self):
        self.temp.cleanup()

    def hdf(self, name="test", n=4, steps=True):
        p = self.root / "mlip" / self.structure / f"nvt_{name}.h5"
        p.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(p, "w") as f:
            f.create_dataset("data/positions", data=np.zeros((n, 1, 3)))
            f.create_dataset("data/stress", data=np.repeat(np.diag([1., 2., 3.])[None], n, axis=0))
            if steps:
                f.create_dataset("steps/positions", data=np.arange(n)*2)
                f.create_dataset("steps/stress", data=np.arange(n)*2)
        return p

    def test_units_sign_shear_and_2d(self):
        stress = [1, 2, 6, 4, 5, 3]
        row = pipeline.stress_row(stress, "bulkCu_300K_test")
        self.assertAlmostEqual(row["pressure_GPa"], -3*pipeline.EV_A3_TO_GPA)
        self.assertEqual(row["stress_xy_eV_A3"], 3)
        self.assertAlmostEqual(pipeline.stress_row(stress, "TiSe2_300K")["pressure_GPa"], -1.5*pipeline.EV_A3_TO_GPA)
        with self.assertRaises(ValueError): pipeline.stress_matrix([np.nan]*6)

    def test_full_stress_tensor_is_symmetrized_without_changing_pressure(self):
        stress = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])

        result = pipeline.stress_matrix(stress)
        row = pipeline.stress_row(stress, "bulkCu_300K_test")

        np.testing.assert_allclose(result, 0.5 * (stress + stress.T))
        self.assertAlmostEqual(
            row["pressure_GPa"],
            -np.trace(stress) / 3 * pipeline.EV_A3_TO_GPA,
        )

    def test_hdf_step_alignment_and_stride(self):
        p = self.hdf()
        df = pipeline.read_stress_frames(p, self.meta)
        np.testing.assert_equal(df.time_fs.to_numpy(), [0, 4, 8, 12])
        with h5py.File(p, "a") as f: f["steps/stress"][2] = 3
        with self.assertRaisesRegex(ValueError, "step numbers"):
            pipeline.read_stress_frames(p, self.meta)

    def test_incomplete_stress_rejected(self):
        p = self.hdf()
        with h5py.File(p, "a") as f:
            del f["data/stress"]
            f.create_dataset("data/stress", data=np.zeros((2, 3, 3)))
        with self.assertRaisesRegex(ValueError, "different frame counts"):
            pipeline.read_stress_frames(p, self.meta)

    def test_shared_time_window_uses_times(self):
        ref = pd.DataFrame({"time_fs": [0, 2, 4, 6, 8], "pressure_GPa": [0]*5})
        model = pd.DataFrame({"time_fs": [0, 4, 8, 12], "pressure_GPa": [0]*4})
        r, m, start, end = pipeline.match_time_window(ref, model)
        self.assertEqual((len(r), len(m), start, end), (5, 3, 0, 8))

    def test_complete_pipeline_and_matched_reference_rescoring(self):
        self.hdf(n=4)
        interface = self.root / "mlip/Pt111w24H2O_380K_Heenen_VASP/nvt_test.h5"
        interface.parent.mkdir()
        interface.touch()  # Exclude it before reading frames or looking up metadata.
        metadata = self.root / "metadata.json"
        metadata.write_text(json.dumps({self.structure: self.meta}))
        ref_path = self.root / "ref" / self.structure / "traj.extxyz"
        ref_path.parent.mkdir(parents=True)
        frames = []
        for i in range(3):
            atoms = Atoms("Cu", positions=[[0,0,0]], cell=[3,3,3], pbc=True)
            atoms.calc = SinglePointCalculator(atoms, energy=0, forces=np.zeros((1,3)), stress=[1,2,3,0,0,0])
            frames.append(atoms)
        write(ref_path, frames)
        output = self.root / "out"
        args = Namespace(metadata=metadata, output_dir=output, reference_file=None,
                         traj_dir=[self.root/"mlip"], prefix=None, model=None,
                         ref_dir=self.root/"ref", reference_first_step=0,
                         bins=8, no_plots=False)
        pairs = pipeline.run_pipeline(args)
        self.assertEqual(pairs.n_mlip_frames.iloc[0], 3)
        self.assertEqual(len(pd.read_csv(output/"test_stress_per_frame.csv")), 4)
        self.assertAlmostEqual(pairs.absolute_mean_error_GPa.iloc[0], 0)
        self.assertAlmostEqual(pairs.pressure_similarity.iloc[0], 1)
        self.assertTrue((output/"plots/pressure_model_comparison.png").is_file())
        self.assertTrue((output/"plots/test"/f"{self.structure}.png").is_file())
        # A full reference containing an unmatched tail must not change the score.
        full_ref = output/pipeline.REFERENCE_NAME
        ref = pd.read_csv(full_ref)
        tail = ref.iloc[[-1]].copy()
        tail["frame_index"], tail["pressure_GPa"] = 99, 1e6
        pd.concat([ref,tail]).to_csv(full_ref,index=False)
        score = build_pair_rows(output, full_ref, pipeline.SUFFIX, 8)
        self.assertAlmostEqual(score.pressure_similarity.iloc[0], 1)

    def test_old_interface_rows_do_not_affect_metrics_or_failure_penalties(self):
        pairs = pd.DataFrame({
            "system": [self.structure, self.structure, "Pt111w24H2O_380K_Heenen_VASP"],
            "system_type": ["pure metals", "pure metals", "metal-water interfaces"],
            "mlip_model": ["test", "other", "test"],
            "pressure_similarity": [0.8, 0.6, 0.0], "bins": [8] * 3,
        })
        penalized = add_failed_system_penalties(
            pairs, self.root, "mlip-trajs-torchsim-eager", None, set(), 8,
            md_data_dir=self.root,
        )
        self.assertEqual(set(penalized.system), {self.structure})
        self.assertEqual(len(penalized), 2)
        system_means, model_means, type_means = write_metric_outputs(
            pairs, self.root / "pairs.csv", self.root / "systems.csv",
            self.root / "models.csv", self.root / "types.csv", None,
        )
        self.assertEqual(set(pd.read_csv(self.root / "pairs.csv").system), {self.structure})
        self.assertEqual(set(system_means.system), {self.structure})
        self.assertEqual(set(type_means.system_type), {"pure metals"})
        self.assertAlmostEqual(model_means.set_index("model").loc["test",
                              "final_mean_pressure_similarity"], 0.8)

    def test_pressure_mae_filters_interface_in_old_summaries(self):
        pd.DataFrame({
            "trajectory_file": [f"/data/{self.structure}/nvt_test.h5",
                                "/data/Pt111w24H2O_380K_Heenen_VASP/nvt_test.h5"],
            "absolute_mean_error_GPa": [2.0, 1000.0],
        }).to_csv(self.root / "test_same-simulation-length_pressure_trajectory_summary.csv",
                  index=False)
        for excluded in (None, set()):
            comparison = build_pressure_mae_from_trajectory_summaries(
                self.root, excluded_system_types=excluded
            )
            self.assertEqual(comparison.error_GPa.tolist(), [2.0])

    def test_nested_evaluator_outputs_use_sibling_references_and_keep_mode(self):
        results = self.root / "results"
        for mode, pressure_mae in (("md_eager", 1.0), ("md_accelerated", 3.0)):
            output = results / "torchsim" / mode
            references = output / "references"
            references.mkdir(parents=True)
            trajectory = self.root / "trajectories" / self.structure / "nvt_test.h5"
            source = f"mlip-trajs-torchsim-{'eager' if mode == 'md_eager' else 'accelerated'}"
            local_md = self.root / "md-data" / source / self.structure
            local_md.mkdir(parents=True)
            (local_md / "nvt_test.h5").touch()
            (local_md / "md_timing_test.csv").write_text(
                f"calculator,system,n_steps\ntest,{self.structure},100\n"
            )
            model_file = output / f"test{pipeline.SUFFIX}"
            pd.DataFrame(
                {
                    "trajectory_file": [str(trajectory)] * 3,
                    "frame_index": [0, 1, 2],
                    "pressure_GPa": [0.0, 1.0, 2.0],
                }
            ).to_csv(model_file, index=False)
            pd.DataFrame(
                {
                    "trajectory_file": [
                        str(self.root / "reference" / self.structure / "traj.extxyz")
                    ] * 3,
                    "frame_index": [0, 1, 2],
                    "pressure_GPa": [0.0, 1.0, 2.0],
                }
            ).to_csv(references / "test.csv", index=False)
            pd.DataFrame(
                {
                    "trajectory_file": [str(trajectory)],
                    "absolute_mean_error_GPa": [pressure_mae],
                }
            ).to_csv(
                output
                / "test_same-simulation-length_pressure_trajectory_summary.csv",
                index=False,
            )

        pairs = build_pair_rows(results, None, pipeline.SUFFIX, 8)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(set(pairs.backend), {"torchsim"})
        self.assertEqual(set(pairs["mode"]), {"md_eager", "md_accelerated"})
        self.assertTrue(pairs.reference_file.str.contains("/references/test.csv").all())

        filtered_pairs = build_pair_rows(
            results, None, pipeline.SUFFIX, 8, models={"TEST"}
        )
        self.assertEqual(len(filtered_pairs), 2)
        self.assertEqual(set(filtered_pairs.mlip_model), {"test"})
        alias_filtered_pairs = build_pair_rows(
            results, None, pipeline.SUFFIX, 8, models={"test-force-only"}
        )
        self.assertEqual(len(alias_filtered_pairs), 2)
        resolved_reference = resolve_reference_pressure_file(
            results / "torchsim" / "md_eager", None
        )
        self.assertEqual(
            resolved_reference,
            results / "torchsim" / "md_eager" / "references" / "test.csv",
        )
        with self.assertRaisesRegex(FileNotFoundError, "missing-model"):
            build_pair_rows(
                results, None, pipeline.SUFFIX, 8, models={"missing-model"}
            )

        _, model_means, _ = write_metric_outputs(
            pairs,
            self.root / "pairs.csv",
            self.root / "system-model.csv",
            self.root / "models.csv",
            self.root / "model-system-type.csv",
            None,
        )
        self.assertEqual(len(model_means), 2)
        self.assertEqual(set(model_means["mode"]), {"md_eager", "md_accelerated"})

        comparison_file = results / "model_mean_pressure_comparison.csv"
        metric_file = results / "model_pressure_error_metric.csv"
        generated = compute_pressure_metric(
            pressures_dir=results,
            reference_file=None,
            model_file_suffix=pipeline.SUFFIX,
            bins=8,
            pair_output_file=results / "pairs-generated.csv",
            system_model_mean_output_file=results / "system-model-generated.csv",
            model_mean_output_file=metric_file,
            model_system_type_mean_output_file=results / "model-type-generated.csv",
            pressure_comparison_file=comparison_file,
            md_data_dir=self.root / "md-data",
        )
        self.assertTrue(comparison_file.is_file())
        self.assertEqual(
            list(pd.read_csv(comparison_file).columns), ["model", "error_GPa"]
        )
        self.assertTrue(np.allclose(generated["pressure_mae_GPa"], 2.0))


if __name__ == "__main__":
    unittest.main()
