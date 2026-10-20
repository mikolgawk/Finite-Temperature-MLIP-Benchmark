from __future__ import annotations

from dataclasses import replace
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent))

from comparison_utils import (
    RDF_SPEC,
    V2_TO_LEGACY_MODEL,
    VDOS_SPEC,
    MetricSpec,
    compare_source,
)
from model_display_names import display_model_name
import compare_vdos_results


class ComparisonUtilsTests(unittest.TestCase):
    def test_requested_model_mapping_and_exclusions(self) -> None:
        self.assertEqual(V2_TO_LEGACY_MODEL["chgnet-force-only-ase"], "chgnet")
        self.assertEqual(V2_TO_LEGACY_MODEL["nequip-oam-l-force-only"], "nequip")
        self.assertIsNone(V2_TO_LEGACY_MODEL["pet-omat-xl-force-only-eager"])
        for base in ("orb-v3-omat", "orb-v3-direct-omat"):
            for tag in ("-force-only", "-force-only-eager"):
                self.assertIsNone(V2_TO_LEGACY_MODEL[base + tag])

    def test_nequip_uses_l_display_name(self) -> None:
        self.assertEqual(display_model_name("nequip-oam-l-force-only"), "NequIP-OAM-L")

    def test_vdos_legacy_fraction_is_converted_to_percent(self) -> None:
        self.assertEqual(VDOS_SPEC.legacy_scale, 100.0)
        self.assertEqual(RDF_SPEC.legacy_scale, 1.0)

    def test_ase_chgnet_is_compared_to_its_existing_v1_model(self) -> None:
        spec = replace(VDOS_SPEC, current_loader=lambda _: pd.DataFrame({
            "current_model": ["chgnet-force-only-ase"], "system": ["bulkCu_1000K_Kapil"],
            "current_error_percent": [20.0],
        }))
        legacy = pd.DataFrame({
            "legacy_model": ["chgnet"], "system": ["bulkCu_1000K_Kapil"], "legacy_error": [0.3],
        })
        comparison, report = compare_source(spec, "accelerated", Path("unused"), legacy)
        self.assertEqual(comparison.legacy_model.tolist(), ["chgnet"])
        self.assertEqual(comparison.delta_percentage_points.tolist(), [-10.0])
        self.assertEqual(report["excluded_new_models"], [])

    def test_vdos_comparison_defaults_to_refined_results_and_allows_legacy_override(self) -> None:
        with patch.object(sys, "argv", ["compare_vdos_results.py"]):
            args = compare_vdos_results.parse_args()
        self.assertEqual(args.v2_results_root, compare_vdos_results.SCRIPT_DIR.parent / "vdos/results")
        with patch.object(sys, "argv", ["compare_vdos_results.py", "--v2-results-root", "/tmp/vdos_legacy/results"]):
            args = compare_vdos_results.parse_args()
        self.assertEqual(args.v2_results_root, Path("/tmp/vdos_legacy/results"))

    def test_unknown_v2_model_fails_instead_of_being_silently_dropped(self) -> None:
        spec = MetricSpec(
            name="test",
            current_column="current_error_percent",
            legacy_column="legacy_error_percent",
            output_prefix="test",
            legacy_scale=1.0,
            current_loader=lambda _: pd.DataFrame(
                {
                    "current_model": ["unmapped-model"],
                    "system": ["system-a"],
                    "current_error_percent": [1.0],
                }
            ),
            legacy_loader=lambda _: pd.DataFrame(),
        )
        legacy = pd.DataFrame(
            {"legacy_model": ["model"], "system": ["system-a"], "legacy_error": [1.0]}
        )
        with self.assertRaisesRegex(ValueError, "No legacy-model mapping"):
            compare_source(spec, "eager", Path("unused"), legacy)


if __name__ == "__main__":
    unittest.main()
