"""CPU regression tests for joining the generated V2 Pareto inputs."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import pandas as pd


SCRIPT = Path(__file__).with_name(
    "plot-pareto-combined-vdos-rdf-pressure-average-similarity-same-simulation-length.py"
)
SPEC = importlib.util.spec_from_file_location("v2_pareto", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
pareto = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pareto)


class ParetoMetricTests(unittest.TestCase):
    def test_source_paths_point_to_generated_v2_outputs(self):
        timings, rdf, vdos, mode = pareto.source_input_paths(
            "mlip-trajs-torchsim-accelerated"
        )
        self.assertEqual(timings.name, "mlip-trajs-torchsim-accelerated")
        self.assertEqual(rdf.parent.name, "mlip-trajs-torchsim-accelerated")
        self.assertEqual(vdos.parent.name, "mlip-trajs-torchsim-accelerated")
        self.assertEqual(mode, "md_accelerated")

    def test_generated_v2_outputs_are_merged_directly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            timings = root / "timings" / "system"
            timings.mkdir(parents=True)
            pd.DataFrame({"seconds_per_step": [0.002]}).to_csv(
                timings / "md_timing_demo-force-only-eager.csv", index=False
            )

            rdf = root / "rdf.csv"
            pd.DataFrame(
                {"Calculator": ["demo-force-only-eager"], "Mean RDF Error [%]": [10.0]}
            ).to_csv(rdf, index=False)

            vdos = root / "vdos.csv"
            pd.DataFrame(
                {"model": ["demo-force-only-eager"], "vdos_error_percent": [20.0]}
            ).to_csv(vdos, index=False)

            pressure = root / "pressure.csv"
            pd.DataFrame(
                {
                    "model": ["demo"],
                    "backend": ["torchsim"],
                    "mode": ["md_eager"],
                    "final_mean_pressure_error_percent": [30.0],
                    "final_mean_pressure_similarity_percent": [70.0],
                }
            ).to_csv(pressure, index=False)

            merged = pareto.load_and_merge(
                timings_dir=timings.parent,
                combined_metrics_file=None,
                rdf_metrics_file=rdf,
                vdos_metrics_file=vdos,
                pressure_metrics_file=pressure,
                pressure_scale_gpa=10.0,
                clip_pressure_error=True,
                pressure_backend="torchsim",
                pressure_mode="md_eager",
            )

            self.assertEqual(merged["model"].tolist(), ["demo"])
            self.assertAlmostEqual(merged["mean_time_per_step_ms"].iloc[0], 2.0)
            self.assertAlmostEqual(merged["RDF Error [%]"].iloc[0], 10.0)
            self.assertAlmostEqual(merged["VDOS Error [%]"].iloc[0], 20.0)
            self.assertAlmostEqual(merged["Pressure Error [%]"].iloc[0], 30.0)
            self.assertAlmostEqual(merged["Combined Error [%]"].iloc[0], 20.0)


if __name__ == "__main__":
    unittest.main()
