#!/usr/bin/env python3
"""Inventory the example peptide PDBs and enumerate non-wrapping motifs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from benchmarks.utilities.pdb_utils import (
    create_pdb_with_selected_residues,
    create_single_model_pdb,
    get_model_count_and_residues,
)


EXAMPLES = (
    ("6AZG", "GSPLFD", 6, "PLP-10 (cis conformer)"),
    ("1SKL", "RRXXRF", 6, "cyclic hexapeptide RR(Nal)(Nal)RF"),
    ("6BE9", "TkNDTsp", 7, "Design7.1"),
    ("7BTI", "XAWXAXC", 7, "Phalloidin"),
    ("6UCX", "PeVkpEvK", 8, "S2-1 (Wednesday)"),
    ("6AXI", "DLFVPPID", 8, "PLP-2"),
    ("6WI3", "XWtDNnmdK", 9, "macrocycle"),
    ("6XBF", "rrLcPIPE", 8, "NDM1i-1g"),
)

PDB_FILENAMES = {
    pdb_id: ("6xbf_nochanges.pdb" if pdb_id == "6XBF" else f"{pdb_id.lower()}.pdb")
    for pdb_id, *_ in EXAMPLES
}

WEIGHTS = {
    6: "weights/6mer/model_1610_current_weights.pt",
    7: "weights/7mer/model_8811_best_weights.pt",
    8: "weights/8mer/model_7731_best_weights.pt",
    9: "weights/9mer/model_6049_best_weights.pt",
}

PREFERRED_CHAINS = {"7BTI": "X", "6WI3": "F", "6XBF": "I"}


def choose_candidate(
    chains: dict[str, list[tuple[str, str]]],
    length: int,
    preferred_chain: str | None = None,
) -> tuple[str, list[tuple[str, str]], str]:
    if preferred_chain and len(chains.get(preferred_chain, [])) == length:
        return preferred_chain, chains[preferred_chain], "preferred deposited peptide copy"
    exact = [(chain, residues) for chain, residues in chains.items() if len(residues) == length]
    if len(exact) == 1:
        return exact[0][0], exact[0][1], "unique chain matching expected length"
    if len(exact) > 1:
        return exact[0][0], exact[0][1], f"first of {len(exact)} chains matching expected length"
    short = [(chain, residues) for chain, residues in chains.items() if 2 <= len(residues) <= 20]
    if short:
        chain, residues = min(short, key=lambda item: abs(len(item[1]) - length))
        return chain, residues, "nearest short-chain length; manual review required"
    return "", [], "no peptide-length chain found; manual review required"


def write_outputs(repo: Path, output_dir: Path) -> None:
    pdb_dir = repo / "pdbs"
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_rows = []
    case_rows = []
    for pdb_id, sheet_sequence, expected_length, peptide_name in EXAMPLES:
        pdb_path = pdb_dir / PDB_FILENAMES[pdb_id]
        if not pdb_path.exists():
            raise FileNotFoundError(
                f"Missing benchmark input {pdb_path}. "
                "See python/benchmarks/README.md for download instructions."
            )
        models, chains = get_model_count_and_residues(pdb_path)
        chain, residues, selection_note = choose_candidate(
            chains, expected_length, PREFERRED_CHAINS.get(pdb_id)
        )
        run_pdb_path = pdb_path
        if models > 1:
            run_pdb_path = pdb_path.with_name(f"{pdb_path.stem}_model1.pdb")
            create_single_model_pdb(pdb_path, run_pdb_path)
        if pdb_id == "6UCX":
            run_pdb_path = pdb_path.with_name("6ucx_peptide.pdb")
            create_pdb_with_selected_residues(
                pdb_path,
                run_pdb_path,
                chain,
                {number for number, _ in residues},
            )
        observed_sequence = "-".join(residue for _, residue in residues)
        residue_ids = ";".join(number for number, _ in residues)
        residue_names = ";".join(residue for _, residue in residues)
        status = "PASS" if len(residues) == expected_length else "REVIEW"
        inventory_rows.append({
            "pdb_id": pdb_id,
            "peptide_name": peptide_name,
            "sheet_sequence": sheet_sequence,
            "expected_length": expected_length,
            "pdb_file": str(run_pdb_path.relative_to(repo)),
            "model": 1,
            "model_count": models,
            "peptide_chain": chain,
            "observed_length": len(residues),
            "observed_residue_names": observed_sequence,
            "pdb_residue_ids": residue_ids,
            "selection_note": selection_note,
            "qc_status": status,
        })
        if status != "PASS":
            continue
        for motif_length in (2, 3, 4):
            for start_index in range(expected_length - motif_length + 1):
                motif_residues = residues[start_index:start_index + motif_length]
                freeze_ids = [number for number, _ in motif_residues]
                run_id = f"{pdb_id}_L{motif_length}_S{start_index + 1:02d}"
                case_rows.append({
                    "run_id": run_id,
                    "pdb_id": pdb_id,
                    "peptide_name": peptide_name,
                    "peptide_length": expected_length,
                    "peptide_chain": chain,
                    "motif_length": motif_length,
                    "motif_start_index": start_index + 1,
                    "motif_end_index": start_index + motif_length,
                    "freeze_residues": " ".join(freeze_ids),
                    "freeze_residue_names": "-".join(residue for _, residue in motif_residues),
                    "desired_motif_start_position": start_index + 1,
                    "pdb_input": str(run_pdb_path.relative_to(repo)),
                    "weights_load": WEIGHTS[expected_length],
                    "contiguity": "Contiguous",
                    "wraparound": "No",
                    "status": "PLANNED",
                })

    for filename, rows in (("example_peptide_input_qc.csv", inventory_rows),
                           ("example_peptide_contiguous_cases.csv", case_rows)):
        with (output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    # Initial production scope: non-wrapping motifs for the six new peptides.
    # NDM1 already has benchmark results, and the 9-mer is deferred.
    run_rows = [
        row for row in case_rows
        if row["pdb_id"] not in {"6XBF", "6WI3"}
    ]
    with (output_dir / "example_peptide_contiguous_run_cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run_rows[0]))
        writer.writeheader()
        writer.writerows(run_rows)
    if len(run_rows) != 90:
        raise RuntimeError(f"enumerated {len(run_rows)} production cases; expected 90")

    pilot_ids = {
        "6AZG_L2_S01",
        "1SKL_L4_S03",
        "6BE9_L3_S03",
        "7BTI_L2_S06",
        "6UCX_L4_S03",
        "6AXI_L3_S06",
    }
    pilot_rows = [row for row in run_rows if row["run_id"] in pilot_ids]
    with (output_dir / "example_peptide_contiguous_pilot_cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pilot_rows[0]))
        writer.writeheader()
        writer.writerows(pilot_rows)
    if len(pilot_rows) != len(pilot_ids):
        raise RuntimeError(f"found {len(pilot_rows)} pilot cases; expected {len(pilot_ids)}")

    expected = sum(3 * length - 6 for _, _, length, _ in EXAMPLES)
    if len(case_rows) != expected:
        raise RuntimeError(f"enumerated {len(case_rows)} cases; expected {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.repo / "outputs" / "example_peptides"
    write_outputs(args.repo, output_dir)


if __name__ == "__main__":
    main()
