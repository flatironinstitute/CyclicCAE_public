import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


PYTHON_DIR = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(PYTHON_DIR))

from benchmarks.utilities.export_scored_silent import (
    export_structure_from_silent,
    select_structure_score,
)


class SelectStructureScoreTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "description": "case_000001",
                "motif_bbHeavy_RMSD": "0.20",
                "score": "-20.0",
            },
            {
                "description": "case_000002",
                "motif_bbHeavy_RMSD": "0.10",
                "score": "-10.0",
            },
        ]

    def test_selects_lowest_motif_rmsd_by_default(self):
        selected = select_structure_score(self.rows)
        self.assertEqual(selected["description"], "case_000002")

    def test_selects_lowest_score(self):
        selected = select_structure_score(self.rows, rank_by="score")
        self.assertEqual(selected["description"], "case_000001")

    def test_selects_an_explicit_tag(self):
        selected = select_structure_score(self.rows, tag="case_000001")
        self.assertEqual(selected["description"], "case_000001")

    def test_rejects_an_unknown_tag(self):
        with self.assertRaisesRegex(ValueError, "tag not found"):
            select_structure_score(self.rows, tag="missing")


class ExportStructureTests(unittest.TestCase):
    def test_exports_the_selected_peptide_chain(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            silent_path = temporary_path / "case.silent"
            silent_path.write_text(
                "SCORE: score motif_bbHeavy_RMSD description\n"
                "SCORE: -20.0 0.20 case_000001\n"
                "SCORE: -10.0 0.10 case_000002\n",
                encoding="utf-8",
            )

            class FakePose:
                def __init__(self, tag):
                    self.tag = tag
                    self.selected_chain = None

                def pdb_info(self):
                    return types.SimpleNamespace(name=lambda: self.tag)

                def split_by_chain(self, chain):
                    self.selected_chain = chain
                    return self

                def dump_pdb(self, path):
                    Path(path).write_text(f"POSE {self.tag}\n", encoding="utf-8")

            poses = [FakePose("case_000001"), FakePose("case_000002")]
            fake_pyrosetta = types.ModuleType("pyrosetta")
            fake_pyrosetta.init = lambda options: None
            fake_pyrosetta.io = types.SimpleNamespace(
                poses_from_silent=lambda path: iter(poses)
            )

            with patch.dict(sys.modules, {"pyrosetta": fake_pyrosetta}):
                output_path, selected = export_structure_from_silent(silent_path)

            self.assertEqual(selected["description"], "case_000002")
            self.assertEqual(output_path, temporary_path / "case_000002.pdb")
            self.assertEqual(output_path.read_text(), "POSE case_000002\n")
            self.assertEqual(poses[1].selected_chain, 1)


if __name__ == "__main__":
    unittest.main()
