import tempfile
import unittest
from pathlib import Path

import pandas as pd

from compute_mean_rmses_by_system_type import load_all_data


class EnergyForceAggregationTests(unittest.TestCase):
    def test_load_all_data_filters_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            for model in ("model-a", "model-b"):
                pd.DataFrame(
                    {
                        "system": ["bulkCu_300K_test"],
                        "energy_rmse": [1.0],
                        "force_rmse": [2.0],
                    }
                ).to_csv(data_dir / f"rmse-results-all_{model}.csv", index=False)

            result = load_all_data(data_dir, {"model-b"})

            self.assertEqual(set(result["calculator"]), {"model-b"})

    def test_load_all_data_rejects_unknown_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(FileNotFoundError, "missing-model"):
                load_all_data(Path(temporary_directory), {"missing-model"})


if __name__ == "__main__":
    unittest.main()
