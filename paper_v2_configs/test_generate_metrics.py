import unittest

import generate_metrics


class GenerateMetricsTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
