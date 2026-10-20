from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import generate_metrics


class GenerateMetricsTests(unittest.TestCase):
    def test_vdos_defaults_to_refined_pipeline_without_narrowing_batch_sources(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for source in generate_metrics.SOURCES:
                physical = "mlip-trajs-ase-eager" if source == "mlip-trajs-ase" else source
                trajectory = root / "data" / physical / "system" / "nvt_model.h5"
                trajectory.parent.mkdir(parents=True)
                trajectory.touch()
            with patch.object(generate_metrics, "SCRIPT_DIR", root):
                stage = next(stage for stage in generate_metrics.metric_stages() if stage.name == "vdos")
            expected = root / "vdos"
            self.assertEqual(stage.working_directory, expected)
            self.assertEqual(stage.command[2], str(expected / "run_vdos_pipeline.py"))
            self.assertIn("--compute-only", stage.command)
            selected = [stage.command[index + 1]
                        for index, value in enumerate(stage.command) if value == "--source"]
            self.assertEqual(selected, list(generate_metrics.SOURCES))

    def test_model_filter_does_not_create_penalties_for_physically_absent_sources(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            trajectory = root / "data/mlip-trajs-torchsim-eager/system/nvt_model-a.h5"
            trajectory.parent.mkdir(parents=True)
            trajectory.touch()
            # A timing-only source is also absent under the previous default
            # discovery policy and must not acquire phantom all-failure rows.
            timing = root / "data/mlip-trajs-ase-accelerated/system/md_timing_model-a.csv"
            timing.parent.mkdir(parents=True)
            timing.touch()
            with patch.object(generate_metrics, "SCRIPT_DIR", root):
                stage = next(stage for stage in generate_metrics.metric_stages(("model-a",))
                             if stage.name == "vdos")
                selected = [stage.command[index + 1]
                            for index, value in enumerate(stage.command) if value == "--source"]
                self.assertEqual(selected, ["mlip-trajs-torchsim-eager"])
                self.assertEqual(stage.command[-2:], ("--model", "model-a"))
                trajectory.unlink()
                stage = next(stage for stage in generate_metrics.metric_stages(("model-a",))
                             if stage.name == "vdos")
                self.assertNotIn("--source", stage.command)

    def test_vdos_explicit_source_does_not_add_other_cohorts(self) -> None:
        source = "mlip-trajs-ase-accelerated"
        stage = next(stage for stage in generate_metrics.metric_stages(sources=(source,))
                     if stage.name == "vdos")
        self.assertEqual(stage.command.count("--source"), 1)
        self.assertEqual(stage.command[-2:], ("--source", source))

    def test_model_filters_are_forwarded_to_every_stage(self) -> None:
        stages = generate_metrics.metric_stages(("model-a", "model-b"))

        for stage in stages:
            self.assertEqual(
                stage.command[-4:],
                ("--model", "model-a", "--model", "model-b"),
                stage.name,
            )

    def test_source_filters_are_forwarded_to_every_stage(self) -> None:
        source = 'mlip-trajs-torchsim-accelerated'
        for stage in generate_metrics.metric_stages(sources=(source,)):
            self.assertEqual(stage.command[-2:], ('--source', source))

    def test_no_model_filter_preserves_existing_commands(self) -> None:
        for stage in generate_metrics.metric_stages():
            self.assertNotIn("--model", stage.command, stage.name)

    def test_molecular_crystals_are_excluded_from_every_stage(self) -> None:
        for stage in generate_metrics.metric_stages():
            exclusion = stage.command.index("--exclude-system-type")
            self.assertEqual(stage.command[exclusion + 1], "molecular crystals")

    def test_molecular_crystal_opt_in_reaches_every_stage(self) -> None:
        for stage in generate_metrics.metric_stages(include_molecular_crystals=True):
            self.assertIn("--include-molecular-crystals", stage.command)
            self.assertNotIn("--exclude-system-type", stage.command)


if __name__ == "__main__":
    unittest.main()
