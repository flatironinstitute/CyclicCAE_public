import csv
import sys
import tempfile
import unittest
from pathlib import Path


PYTHON_DIR = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(PYTHON_DIR))

from benchmarks.utilities.summary_utils import upsert_benchmark_summaries


class UpsertBenchmarkSummariesTests(unittest.TestCase):
    def test_inserts_and_replaces_rows_by_permutation_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            upsert_benchmark_summaries(
                path,
                [
                    {
                        "peptide_name": "6XBF_L2_S01",
                        "permutation_name": "6XBF_L2_S01",
                        "status": "FAIL",
                    },
                    {
                        "peptide_name": "6AZG_L2_S01",
                        "permutation_name": "6AZG_L2_S01",
                        "status": "PASS",
                    },
                ],
            )
            upsert_benchmark_summaries(
                path,
                [
                    {
                        "peptide_name": "6XBF_L2_S01",
                        "permutation_name": "6XBF_L2_S01",
                        "status": "PASS",
                    }
                ],
            )

            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["permutation_name"], "6XBF_L2_S01")
            self.assertEqual(rows[0]["status"], "PASS")
            self.assertEqual(rows[1]["permutation_name"], "6AZG_L2_S01")


if __name__ == "__main__":
    unittest.main()
