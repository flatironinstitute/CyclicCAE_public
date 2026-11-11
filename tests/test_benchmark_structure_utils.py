import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PYTHON_DIR = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(PYTHON_DIR))

from benchmarks.utilities.structure_utils import download_pdb_structure


PDB_CONTENT = b"ATOM      1  CA  GLY A   1       0.000   0.000   0.000\nEND\n"


class DownloadPdbStructureTests(unittest.TestCase):
    def test_downloads_and_validates_structure(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "1abc.pdb"
            with patch(
                "benchmarks.utilities.structure_utils.urlopen",
                return_value=io.BytesIO(PDB_CONTENT),
            ) as mocked_open:
                downloaded = download_pdb_structure("1abc", destination)

            self.assertTrue(downloaded)
            self.assertEqual(destination.read_bytes(), PDB_CONTENT)
            request = mocked_open.call_args.args[0]
            self.assertEqual(
                request.full_url,
                "https://files.rcsb.org/download/1ABC.pdb",
            )

    def test_keeps_an_existing_valid_structure(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "1abc.pdb"
            destination.write_bytes(PDB_CONTENT)
            with patch("benchmarks.utilities.structure_utils.urlopen") as mocked_open:
                downloaded = download_pdb_structure("1abc", destination)

            self.assertFalse(downloaded)
            mocked_open.assert_not_called()

    def test_rejects_a_response_without_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "1abc.pdb"
            with patch(
                "benchmarks.utilities.structure_utils.urlopen",
                return_value=io.BytesIO(b"not a PDB\n"),
            ):
                with self.assertRaises(ValueError):
                    download_pdb_structure("1abc", destination)

            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(directory).glob("*.part")), [])


if __name__ == "__main__":
    unittest.main()
