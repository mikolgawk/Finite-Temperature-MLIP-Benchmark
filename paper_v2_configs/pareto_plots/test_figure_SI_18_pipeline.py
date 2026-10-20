"""Check Figure SI 18 commands without running any plot scripts."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import plot_all


TORCHSIM = "mlip-trajs-torchsim-accelerated"
ASE = "mlip-trajs-ase-accelerated"
COMPARISON_DIR = "torchsim_vs_ase_accelerated"


class FigureSI18PipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def run_pipeline(self, sources=None, extra_args=(), available=()):
        argv = ["plot_all.py", "--plots-dir", str(self.root / "plots"),
                "--results-dir", str(self.root / "results"), *extra_args]
        for source in sources or ():
            argv.extend(["--source", source])
        with (patch.object(sys, "argv", argv),
              patch.object(plot_all, "available_sources", return_value=list(available)),
              patch.object(plot_all, "comparison_available", return_value=False),
              patch.object(plot_all, "require_vdos_metrics_file"),
              patch.object(plot_all, "run_stage", return_value=True) as run_stage,
              contextlib.redirect_stdout(io.StringIO())):
            plot_all.main()
        self.assertFalse((self.root / "plots").exists())
        self.assertFalse((self.root / "results").exists())
        self.all_commands = [call.args[1] for call in run_stage.call_args_list]
        return [command for command in self.all_commands
                if Path(command[2]).name in {"figure_7.py", "figure_SI_18.py"}]

    @staticmethod
    def option_values(command, option):
        return [command[index + 1] for index, argument in enumerate(command) if argument == option]

    def combined_commands(self, commands):
        return [command for command in commands if Path(command[2]).name == "figure_SI_18.py"]

    def frontier_path(self, command):
        explicit = self.option_values(command, "--frontier-csv")
        if explicit:
            return Path(explicit[0])
        return Path(self.option_values(command, "--output-csv")[0]).with_name("figure_SI_18_frontier.csv")

    def test_selecting_both_accelerated_runners_adds_one_combined_plot(self):
        commands = self.run_pipeline([ASE, TORCHSIM, ASE, TORCHSIM])
        self.assertEqual(len(commands), 2)
        combined = self.combined_commands(commands)
        self.assertEqual(len(combined), 1)
        command = combined[0]
        self.assertEqual(self.option_values(command, "--source"), [TORCHSIM, ASE])
        self.assertNotIn("--pressure-backend", command)
        self.assertNotIn("--pressure-mode", command)
        self.assertEqual(self.option_values(command, "--output-file"),
                         [str(self.root / "plots" / COMPARISON_DIR / "figure_SI_18.pdf")])
        self.assertEqual(self.option_values(command, "--output-csv"),
                         [str(self.root / "results" / COMPARISON_DIR / "figure_SI_18_metrics.csv")])
        self.assertEqual(self.frontier_path(command),
                         self.root / "results" / COMPARISON_DIR / "figure_SI_18_frontier.csv")
        single_sources = [self.option_values(command, "--source")[0]
                          for command in commands if Path(command[2]).name == "figure_7.py"]
        self.assertEqual(single_sources, [TORCHSIM])

    def test_a_single_selected_accelerated_runner_does_not_add_comparison(self):
        selections = ([TORCHSIM], [ASE], [ASE, "mlip-trajs-torchsim-eager"],
                      [TORCHSIM, "mlip-trajs-ase"])
        for sources in selections:
            with self.subTest(sources=sources):
                commands = self.run_pipeline(sources)
                expected_sources = [source for source in sources if "torchsim" in source]
                self.assertEqual(len(commands), len(expected_sources))
                self.assertEqual(self.combined_commands(commands), [])
                self.assertEqual([self.option_values(command, "--source")[0] for command in commands],
                                 expected_sources)

    def test_automatic_available_sources_also_include_combined_plot(self):
        commands = self.run_pipeline(available=[TORCHSIM, ASE])
        self.assertEqual(len(commands), 2)
        self.assertEqual(len(self.combined_commands(commands)), 1)

    def test_combined_paths_and_shared_pressure_override_respect_custom_roots(self):
        plots = self.root / "custom plot directory"
        results = self.root / "custom results directory"
        pressure = self.root / "shared pressure.csv"
        commands = self.run_pipeline([TORCHSIM, ASE], extra_args=(
            "--plots-dir", str(plots), "--results-dir", str(results),
            "--pressure-file", str(pressure), "--include-molecular-crystals",
        ))
        command = self.combined_commands(commands)[0]
        self.assertEqual(self.option_values(command, "--output-file"),
                         [str(plots / COMPARISON_DIR / "figure_SI_18.pdf")])
        self.assertEqual(self.option_values(command, "--output-csv"),
                         [str(results / COMPARISON_DIR / "figure_SI_18_metrics.csv")])
        self.assertEqual(self.frontier_path(command),
                         results / COMPARISON_DIR / "figure_SI_18_frontier.csv")
        pressure_values = self.option_values(command, "--pressure-metrics-file")
        self.assertIn(pressure_values, ([str(pressure)], [str(pressure), str(pressure)]))
        self.assertIn("--include-molecular-crystals", command)
        self.assertFalse(plots.exists())
        self.assertFalse(results.exists())

    def test_combined_default_pressure_selection_is_inferred_for_each_source(self):
        commands = self.run_pipeline([TORCHSIM, ASE])
        command = self.combined_commands(commands)[0]
        self.assertEqual(self.option_values(command, "--source"), [TORCHSIM, ASE])
        for option in ("--pressure-metrics-file", "--pressure-backend", "--pressure-mode"):
            self.assertNotIn(option, command)

    def test_figure_7_stays_torchsim_only_and_si_10_to_13_keep_all_selected_sources(self):
        sources = ["mlip-trajs-torchsim-eager", TORCHSIM, ASE]
        commands = self.run_pipeline(sources)
        figure7 = [command for command in commands if Path(command[2]).name == "figure_7.py"]
        self.assertEqual([self.option_values(command, "--source")[0] for command in figure7],
                         sources[:2])
        self.assertEqual(len(self.combined_commands(commands)), 1)
        for number in (10, 11, 12, 13):
            selected = [command for command in self.all_commands
                        if Path(command[2]).name == f"figure_SI_{number}.py"]
            self.assertEqual([self.option_values(command, "--source")[0] for command in selected], sources)

    def test_all_vdos_dependent_commands_receive_explicit_refined_inputs(self):
        commands = self.run_pipeline([TORCHSIM, ASE])
        combined = self.combined_commands(commands)[0]
        root = plot_all.DEFAULT_VDOS_RESULTS_DIR
        self.assertEqual(self.option_values(combined, "--vdos-results-dir"), [str(root)])
        for command in self.all_commands:
            script = Path(command[2]).name
            if script in {"figure_7.py", "figure_SI_11.py", "figure_SI_12.py", "figure_SI_13.py"}:
                source = self.option_values(command, "--source")[0]
                flag = "--vdos-file" if script in {"figure_SI_11.py", "figure_SI_12.py"} else "--vdos-metrics-file"
                self.assertEqual(self.option_values(command, flag),
                                 [str(plot_all.vdos_metrics_path(source, root))])

    def test_explicit_legacy_root_is_forwarded_to_all_comparison_sources(self):
        root = self.root / "vdos_legacy/results"
        commands = self.run_pipeline([TORCHSIM, ASE], extra_args=("--vdos-results-dir", str(root)))
        self.assertEqual(self.option_values(self.combined_commands(commands)[0], "--vdos-results-dir"), [str(root)])
        for command in self.all_commands:
            for flag in ("--vdos-file", "--vdos-metrics-file"):
                values = self.option_values(command, flag)
                if values:
                    source = self.option_values(command, "--source")[0]
                    self.assertEqual(values, [str(plot_all.vdos_metrics_path(source, root))])


if __name__ == "__main__":
    unittest.main()
