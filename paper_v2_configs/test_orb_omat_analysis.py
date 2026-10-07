"""Check OMAT evaluator settings and keep checkpoint/source results separate."""
from __future__ import annotations

import ast
import contextlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from model_display_names import display_model_name, normalize_display_key

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pressures"))
sys.path.insert(0, str(ROOT / "e_f_rmses"))
from pressure_evaluator import _trajectory_model, _trajectory_paths
from compute_mean_rmses_by_system_type import load_all_data


def md_settings(mode: str, suffix: str) -> dict:
    """Read the authoritative loader/numeric settings without running MD."""
    tree = ast.parse((ROOT / mode / "torchsim-scripts" / f"md_{suffix}.py").read_text())
    assignments = {
        ast.unparse(target): node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
    }
    loader = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and ast.unparse(node.func).startswith("pretrained.")
    )
    keywords = {keyword.arg: keyword.value for keyword in loader.keywords}
    compile_value = keywords["compile"]
    if isinstance(compile_value, ast.Name):
        compile_value = assignments[compile_value.id]
    matmul = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "torch.set_float32_matmul_precision"
    )
    return {
        "loader": loader.func.attr,
        "precision": ast.literal_eval(keywords["precision"]),
        "compile": ast.literal_eval(compile_value),
        "matmul": ast.literal_eval(matmul.args[0]),
        "tf32": ast.literal_eval(assignments["torch.backends.cuda.matmul.allow_tf32"]),
    }


class OrbOmatAnalysisTests(unittest.TestCase):
    def test_v2_orb_v3_runners_and_registries_use_only_omat(self):
        runners = list(ROOT.rglob("*orb_v3*.py"))
        self.assertEqual(len(runners), 24)
        for runner in runners:
            self.assertIn("_omat", runner.stem, str(runner))
            for node in ast.walk(ast.parse(runner.read_text())):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr.startswith("orb_v3")):
                    self.assertIn("_omat", node.func.attr, str(runner))
        for relative in ("md_eager/model_calculators.json", "md_accelerated/model_calculators.json",
                         "e_f_rmses/model_calculators.json"):
            registry = json.loads((ROOT / relative).read_text())
            entries = [entry for entry in registry["models"] if entry["name"].startswith("orb-v3")]
            self.assertEqual({entry["name"] for entry in entries},
                             {"orb-v3-omat", "orb-v3-direct-omat"})
            for entry in entries:
                expression = entry["calculator_expr"]
                ast.parse(expression, mode="eval")
                self.assertIn("_omat(", expression)
        registry = json.loads((ROOT / "md_eager/md_calculator_versions.json").read_text())
        entries = {key: value for key, value in registry["calculators"].items() if key.startswith("orb-v3")}
        self.assertEqual(set(entries), {"orb-v3-omat", "orb-v3-direct-omat"})
        for entry in entries.values():
            self.assertIn("_omat", entry["model"]["identifier"])
            self.assertTrue((ROOT / "md_eager/torchsim-scripts" / entry["script"]).is_file())

    def test_all_evaluators_match_production_checkpoint_and_numeric_settings(self):
        for mode in ("md_eager", "md_accelerated"):
            eager = mode == "md_eager"
            for suffix, base in (
                ("orb_v3_omat", "orb-v3-omat"),
                ("orb_v3_direct_omat", "orb-v3-direct-omat"),
            ):
                settings = md_settings(mode, suffix)
                for metric in ("rmse", "pressure"):
                    with self.subTest(mode=mode, model=base, metric=metric):
                        field = Mock()
                        field.named_modules.return_value = []
                        pretrained = SimpleNamespace(**{
                            settings["loader"]: Mock(return_value=(field, "adapter"))
                        })
                        torch = SimpleNamespace(
                            device=Mock(return_value="cuda"), float32="float32", dtype=object,
                            set_float32_matmul_precision=Mock(),
                            backends=SimpleNamespace(
                                cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=None)),
                                cudnn=SimpleNamespace(allow_tf32=None, benchmark=None),
                            ),
                            jit=SimpleNamespace(ScriptModule=type("ScriptModule", (), {})),
                        )
                        adapter = Mock(side_effect=lambda model, *args, **kwargs:
                                       SimpleNamespace(compute_stress=not model.disable_stress.called))
                        helper = SimpleNamespace(
                            early_cli=Mock(), run_rmse=Mock(), run_torchsim_pressure=Mock()
                        )
                        modules = {
                            "torch": torch,
                            "torch._dynamo.eval_frame": SimpleNamespace(
                                OptimizedModule=type("OptimizedModule", (), {})),
                            "orb_models": SimpleNamespace(),
                            "orb_models.forcefield": SimpleNamespace(pretrained=pretrained),
                            "torch_sim": SimpleNamespace(),
                            "torch_sim.models": SimpleNamespace(),
                            "torch_sim.models.orb": SimpleNamespace(OrbModel=adapter),
                            "torchsim_rmse": helper, "pressure_evaluator": helper,
                            "ase.io": SimpleNamespace(read=Mock()),
                        }
                        if metric == "rmse":
                            directory = ROOT / "e_f_rmses/rmse_torchsim_scripts" / (
                                "md_eager" if eager else "md-accelerated")
                            evaluate = helper.run_rmse
                            model_name = base + "-force-only" + ("-eager" if eager else "")
                        else:
                            directory = ROOT / "pressures" / mode / "torchsim-scripts"
                            evaluate = helper.run_torchsim_pressure
                            model_name = base + "-stress" + ("-eager" if eager else "")
                        script = directory / f"{metric}_{suffix}.py"
                        with patch.dict(sys.modules, modules), patch.object(sys, "path", sys.path.copy()):
                            runpy.run_path(str(script), run_name="__main__")
                        getattr(pretrained, settings["loader"]).assert_called_once_with(
                            device="cuda", precision=settings["precision"], compile=settings["compile"])
                        torch.set_float32_matmul_precision.assert_called_once_with(settings["matmul"])
                        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, settings["tf32"])
                        self.assertFalse(torch.backends.cudnn.allow_tf32)
                        self.assertFalse(torch.backends.cudnn.benchmark)
                        self.assertEqual(field.disable_stress.call_count, int(metric == "rmse"))
                        evaluate.assert_called_once()
                        self.assertEqual(evaluate.call_args.args[0], model_name)
                        self.assertEqual(evaluate.call_args.args[1].compute_stress, metric == "pressure")
                        self.assertEqual(evaluate.call_args.kwargs.get("state_dtype", "float32"), "float32")
                        self.assertEqual(adapter.call_args.kwargs.get("dtype", "float32"), "float32")
                        helper.early_cli.assert_called_once()
                        self.assertEqual(Path(helper.early_cli.call_args.args[0]), script)

    def test_ase_evaluators_match_checkpoint_compile_and_stress_settings(self):
        for mode in ("md_eager", "md_accelerated"):
            eager = mode == "md_eager"
            for suffix, base in (("orb_v3_omat", "orb-v3-omat"),
                                 ("orb_v3_direct_omat", "orb-v3-direct-omat")):
                settings = md_settings(mode, suffix)
                for metric in ("md", "rmse", "pressure"):
                    with self.subTest(mode=mode, model=base, metric=metric):
                        field = Mock()
                        field.named_modules.return_value = []
                        loader = Mock(return_value=(field, "adapter"))
                        pretrained = SimpleNamespace(**{settings["loader"]: loader})
                        torch = SimpleNamespace(
                            set_float32_matmul_precision=Mock(),
                            backends=SimpleNamespace(
                                cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=None)),
                                cudnn=SimpleNamespace(allow_tf32=None, benchmark=None)),
                            jit=SimpleNamespace(ScriptModule=type("ScriptModule", (), {})),
                        )
                        calculator = SimpleNamespace(implemented_properties=["energy", "forces", "stress"])
                        adapter = Mock(return_value=calculator)
                        evaluate = Mock(side_effect=lambda name, factory, *args, **kwargs: factory())
                        helper = SimpleNamespace(REPO=ROOT.parent, early_cli=Mock(),
                                                 run_rmse=evaluate, run_ase_pressure=evaluate)
                        modules = {
                            "torch": torch,
                            "torch._dynamo.eval_frame": SimpleNamespace(
                                OptimizedModule=type("OptimizedModule", (), {})),
                            "orb_models": SimpleNamespace(),
                            "orb_models.forcefield": SimpleNamespace(pretrained=pretrained),
                            "orb_models.forcefield.inference.calculator": SimpleNamespace(ORBCalculator=adapter),
                            "ase": SimpleNamespace(units=SimpleNamespace()),
                            "ase.io": SimpleNamespace(read=Mock()),
                            "ase.md": SimpleNamespace(MDLogger=Mock()),
                            "ase.md.bussi": SimpleNamespace(Bussi=Mock()),
                            "ase.md.langevin": SimpleNamespace(Langevin=Mock()),
                            "ase.md.nose_hoover_chain": SimpleNamespace(NoseHooverChainNVT=Mock()),
                            "ase.md.velocitydistribution": SimpleNamespace(MaxwellBoltzmannDistribution=Mock()),
                            "ase_rmse": helper, "pressure_evaluator": helper,
                            "_ase_md": SimpleNamespace(run_ase_md=evaluate),
                        }
                        if metric == "md":
                            directory = ROOT / mode / "ase-scripts"
                        elif metric == "rmse":
                            directory = ROOT / "e_f_rmses/rmse_ase_scripts" / (
                                "md_eager" if eager else "md-accelerated")
                        else:
                            directory = ROOT / "pressures" / mode / "ase-scripts"
                        script = directory / f"{metric}_{suffix}.py"
                        with patch.dict(sys.modules, modules), patch.object(sys, "path", sys.path.copy()):
                            if metric == "md" and eager:
                                namespace = runpy.run_path(str(script))
                                namespace["make_calculator"]()
                            else:
                                runpy.run_path(str(script), run_name="__main__")
                        loader.assert_called_once_with(device="cuda", precision=settings["precision"],
                                                       compile=settings["compile"])
                        adapter.assert_called_once_with(field, atoms_adapter="adapter", device="cuda")
                        self.assertEqual(field.disable_stress.call_count, int(metric != "pressure"))
                        expected_name = base + ("" if eager else "-stress" if metric == "pressure" else "-force-only")
                        if metric != "md" or not eager:
                            self.assertEqual(evaluate.call_args.args[0], expected_name)
                        if metric != "md":
                            helper.early_cli.assert_called_once()
                        if metric == "pressure":
                            source = "mlip-trajs-ase" + ("" if eager else "-accelerated")
                            self.assertEqual(helper.early_cli.call_args.kwargs["default_traj_dir"],
                                             ROOT / "data" / source)
                        if not eager or metric == "md":
                            self.assertEqual("stress" in calculator.implemented_properties, metric == "pressure")
                        torch.set_float32_matmul_precision.assert_called_once_with(settings["matmul"])
                        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, settings["tf32"])
                        self.assertFalse(torch.backends.cudnn.allow_tf32)
                        self.assertFalse(torch.backends.cudnn.benchmark)

    def test_ase_omat_md_outputs_feed_metric_trajectory_lookup(self):
        try:
            import ase.md.bussi
            import ase.md.nose_hoover_chain
        except ImportError:
            self.skipTest("ASE >=3.26 is required for the production MD driver")
        import h5py
        import numpy as np
        from ase.build import bulk
        from ase.calculators.emt import EMT
        from ase.io import write

        class EnergyForceOnlyEMT(EMT):
            implemented_properties = ["energy", "free_energy", "forces"]

        for mode in ("md_eager", "md_accelerated"):
            eager = mode == "md_eager"
            for base in ("orb-v3-omat", "orb-v3-direct-omat"):
                with self.subTest(mode=mode, model=base), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    system = "bulkCu_300K_test"
                    write(root / "init.extxyz", bulk("Cu", "fcc", a=3.6, cubic=True))
                    metadata = root / "metadata.json"
                    metadata.write_text(json.dumps({system: {
                        "initfile_path": "init.extxyz", "temperature": 300,
                        "timestep": 1, "thermostat_coupling_constant": 50,
                        "thermostat_type": "Langevin", "position_print_stride": 1,
                        "energy_print_stride": 1, "trajectory_length_ps": 0.002,
                    }}))
                    directory = ROOT / mode / "ase-scripts"
                    with patch.object(sys, "path", [str(directory), *sys.path]):
                        namespace = runpy.run_path(str(directory / f"md_{base.replace('-', '_')}.py"))
                    driver = namespace["run_ase_md"]
                    source = "mlip-trajs-ase" + ("" if eager else "-accelerated")
                    self.assertEqual(driver.__globals__["OUT_ROOT"], ROOT / "data" / source)
                    model = base + ("" if eager else "-force-only")
                    options = {} if eager else {"filename_suffix": ""}
                    with patch.dict(driver.__globals__, {
                        "REPO": root, "METADATA_FILE": metadata, "OUT_ROOT": root / source,
                    }), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        driver(model, EnergyForceOnlyEMT, "ase-test", synchronize=lambda: None, **options)
                    output = root / source / system
                    trajectory = output / f"nvt_{model}.h5"
                    self.assertEqual(_trajectory_paths(root / source, base + "-stress"), [trajectory])
                    timing = pd.read_csv(output / f"md_timing_{model}.csv")
                    self.assertEqual(timing.calculator.tolist(), [model])
                    self.assertEqual(timing.n_steps.tolist(), [2])
                    with h5py.File(trajectory) as handle:
                        self.assertEqual(handle.attrs["engine"], "ase")
                        self.assertEqual(handle.attrs["model"], model)
                        self.assertEqual(handle["data/positions"].shape, (3, 4, 3))
                        self.assertTrue(np.isfinite(handle["data/velocities"][:]).all())
                        self.assertNotIn("stress", handle["data"])

    def test_pressure_selects_only_matching_omat_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = ("orb-v2-force-only-eager", "pet-oam-xl-force-only-eager",
                     "orb-v3-omat-force-only-eager", "orb-v3-direct-omat-force-only-eager")
            for name in names:
                (root / f"nvt_{name}.h5").touch()
            for base in ("orb-v3-omat", "orb-v3-direct-omat"):
                self.assertEqual(_trajectory_model(base + "-stress-eager"), base)
                paths = _trajectory_paths(root, base + "-stress-eager")
                self.assertEqual([path.name for path in paths], [f"nvt_{base}-force-only-eager.h5"])

    def test_rmse_aggregation_keeps_checkpoints_and_sources_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = {}
            for mode, source, offset in (
                ("md_eager", "mlip-trajs-torchsim-eager", 0),
                ("md-accelerated", "mlip-trajs-torchsim-accelerated", 10),
            ):
                predictions = root / "predictions" / mode
                predictions.mkdir(parents=True)
                for value, base in enumerate(("orb-v2", "pet-oam-xl", "orb-v3-omat", "orb-v3-direct-omat"), 1):
                    model = base + "-force-only" + ("-eager" if mode == "md_eager" else "")
                    system = "bulkCu_300K_test"
                    md_dir = root / "data" / source / system
                    md_dir.mkdir(parents=True, exist_ok=True)
                    (md_dir / f"nvt_{model}.h5").touch()
                    (md_dir / f"md_timing_{model}.csv").write_text(
                        f"calculator,system,n_steps\n{model},{system},100\n")
                    pd.DataFrame({
                        "system": ["bulkCu"], "trajectory": [str(root / "ref" / system / "traj.extxyz")],
                        "energy_rmse": [value + offset], "force_rmse": [2 * (value + offset)],
                    }).to_csv(predictions / f"rmse-results-all_{model}.csv", index=False)
                    expected[(source, model)] = value + offset
            result = load_all_data(root / "predictions", md_data_dir=root / "data")
            self.assertEqual({(row.source, row.calculator): row.energy_rmse for row in result.itertuples()}, expected)
            selected = {model for source, model in expected if "omat" in model}
            result = load_all_data(root / "predictions", selected, md_data_dir=root / "data")
            self.assertEqual(set(result.calculator), selected)
            self.assertEqual(len(result), 4)

    def test_display_keys_preserve_omat_checkpoint_identity(self):
        for base, label in (("orb-v3-omat", "orb-v3-conservative-inf-omat"),
                            ("orb-v3-direct-omat", "orb-v3-direct-20-omat")):
            for tag in ("-force-only", "-force-only-eager", "-stress", "-stress-eager"):
                self.assertEqual(normalize_display_key(base + tag), base)
                self.assertEqual(display_model_name(base + tag), label)

    def test_every_new_entry_point_has_lightweight_help(self):
        scripts = [
            *ROOT.glob("e_f_rmses/rmse_torchsim_scripts/*/rmse_orb_v3*omat.py"),
            *ROOT.glob("e_f_rmses/rmse_ase_scripts/*/rmse_orb_v3*omat.py"),
            *ROOT.glob("pressures/*/torchsim-scripts/pressure_orb_v3*omat.py"),
            *ROOT.glob("pressures/*/ase-scripts/pressure_orb_v3*omat.py"),
        ]
        self.assertEqual(len(scripts), 16)
        for script in scripts:
            with self.subTest(script=script.relative_to(ROOT)):
                result = subprocess.run([sys.executable, str(script), "--help"],
                                        capture_output=True, text=True, check=True)
                self.assertIn("--max-frames", result.stdout)
                self.assertIn("--output-dir", result.stdout)


if __name__ == "__main__":
    unittest.main()
