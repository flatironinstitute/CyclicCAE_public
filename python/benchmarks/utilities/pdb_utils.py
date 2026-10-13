"""PDB inspection and extraction helpers for benchmark preparation."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path


def get_model_count_and_residues(
    path: Path,
) -> tuple[int, dict[str, list[tuple[str, str]]]]:
    """Return the model count and ordered polymer-like residues in model one."""
    chains: dict[str, OrderedDict[tuple[str, str], tuple[str, set[str]]]] = {}
    model_count = 0
    in_first_model = True
    saw_model = False
    with path.open(errors="replace") as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "MODEL":
                model_count += 1
                saw_model = True
                in_first_model = model_count == 1
                continue
            if record == "ENDMDL" and in_first_model:
                in_first_model = False
                continue
            if record not in {"ATOM", "HETATM"} or not in_first_model:
                continue
            atom_name = line[12:16].strip()
            if atom_name not in {"N", "CA", "C", "O"}:
                continue
            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                continue
            chain = line[21].strip() or "_"
            key = (line[22:26].strip(), line[26].strip())
            residue_name = line[17:20].strip()
            residues = chains.setdefault(chain, OrderedDict())
            if key not in residues:
                residues[key] = (residue_name, set())
            residues[key][1].add(atom_name)
    if not saw_model:
        model_count = 1
    return model_count, {
        chain: [
            (f"{number}{insertion_code}", residue_name)
            for (number, insertion_code), (residue_name, atoms) in residues.items()
            if {"N", "CA", "C"}.issubset(atoms)
        ]
        for chain, residues in chains.items()
    }


def create_single_model_pdb(source: Path, destination: Path) -> None:
    """Create a PDB containing headers and coordinates from the first model."""
    output = []
    saw_model = False
    in_first_model = False
    with source.open(errors="replace") as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "MODEL":
                if saw_model:
                    break
                saw_model = True
                in_first_model = True
                continue
            if record == "ENDMDL" and in_first_model:
                break
            if not saw_model or in_first_model:
                output.append(line)
    if not output or not saw_model:
        raise ValueError(f"{source} does not contain a MODEL record")
    output.append("END\n")
    destination.write_text("".join(output))


def create_pdb_with_selected_residues(
    source: Path,
    destination: Path,
    chain: str,
    residue_ids: set[str],
) -> None:
    """Create a PDB containing only selected coordinates and safe headers."""
    output = []
    with source.open(errors="replace") as handle:
        for line in handle:
            record = line[:6].strip()
            if record in {"ATOM", "HETATM", "TER"}:
                line_chain = line[21].strip() or "_"
                residue_id = f"{line[22:26].strip()}{line[26].strip()}"
                if line_chain == chain and residue_id in residue_ids:
                    output.append(line)
            elif record not in {"CONECT", "MASTER", "END"}:
                output.append(line)
    output.append("END\n")
    destination.write_text("".join(output))
