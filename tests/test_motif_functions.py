import sys
import unittest
from pathlib import Path


PYTHON_DIR = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(PYTHON_DIR))

import torch

from motif_functions import calculate_motif_numbering_for_kabsch


class CalculateMotifNumberingTests(unittest.TestCase):
    def test_converts_one_based_input_positions(self):
        motif, motif_input = calculate_motif_numbering_for_kabsch(
            torch.tensor([[0, 2]]),
            pep_len=8,
            motif_length=2,
            motif_positions=torch.tensor([3, 4]),
        )

        self.assertEqual(motif.tolist(), [list(range(8, 16))])
        self.assertEqual(motif_input.tolist(), list(range(8, 16)))

    def test_wraps_output_motif_across_peptide_terminus(self):
        motif, motif_input = calculate_motif_numbering_for_kabsch(
            torch.tensor([[0, 7]]),
            pep_len=8,
            motif_length=2,
            motif_positions=torch.tensor([8, 1]),
        )

        expected = [28, 29, 30, 31, 0, 1, 2, 3]
        self.assertEqual(motif.tolist(), [expected])
        self.assertEqual(motif_input.tolist(), expected)

    def test_rejects_positions_outside_peptide(self):
        for positions in (torch.tensor([0, 1]), torch.tensor([8, 9])):
            with self.subTest(positions=positions.tolist()):
                with self.assertRaises(IndexError):
                    calculate_motif_numbering_for_kabsch(
                        torch.tensor([[0, 0]]),
                        pep_len=8,
                        motif_length=2,
                        motif_positions=positions,
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_keeps_generated_indices_on_cuda(self):
        device = torch.device("cuda")
        motif, motif_input = calculate_motif_numbering_for_kabsch(
            torch.tensor([[0, 7]], device=device),
            pep_len=8,
            motif_length=2,
            motif_positions=torch.tensor([8, 1], device=device),
        )

        self.assertEqual(motif.device.type, "cuda")
        self.assertEqual(motif_input.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
